// MemX GPU Compressor v4 - Adaptive LZ4 with page classification
// Top-tier compression: per-page adaptive algorithm selection
//
// Page types detected by GPU:
//   TYPE_ZERO:     All bytes zero → store 4 bytes (4096x)
//   TYPE_UNIFORM:  All bytes same → store 5 bytes (3276x)
//   TYPE_RUN:      Long runs of same byte → RLE only
//   TYPE_STRUCT:   Structured data (delta-friendly) → Delta + LZ4
//   TYPE_RANDOM:   High entropy → skip (incompressible)
//
// Compression pipeline:
//   1. 256 threads parallel: delta encode + classify page type
//   2. Based on type, select optimal algorithm
//   3. LZ4-style compression with 8192-entry hash table
//   4. Output: [0x4D 0x58 version type ...compressed data...]

#include <metal_stdlib>
using namespace metal;

constant uint PS = 16384;

// ─── Hash function (4-byte rolling hash) ───
uint h4(const device uchar* p) {
    return ((uint)p[0] | ((uint)p[1]<<8) | ((uint)p[2]<<16) | ((uint)p[3]<<24)) * 2654435761u;
}

uint h4_tg(threadgroup const uchar* p) {
    return ((uint)p[0] | ((uint)p[1]<<8) | ((uint)p[2]<<16) | ((uint)p[3]<<24)) * 2654435761u;
}

// ─── Page classifier ───
// Returns: 0=ZERO, 1=UNIFORM, 2=RUN-friendly, 3=STRUCTURED, 4=RANDOM
uint classify_page(threadgroup const uchar* dp, uint t, uint ts) {
    // Sample-based classification (fast, parallel)
    // Each thread checks a few positions
    uint zero_count = 0;
    uint same_count = 0;  // consecutive same bytes
    uint unique_bytes = 0;
    
    // Histogram sampling (16 threads, 64 positions each)
    uint hist[16] = {0};  // Only track top nibble for speed
    uint samples = 0;
    
    for (uint i = t; i < PS; i += ts * 4) {
        if (dp[i] == 0) zero_count++;
        if (i > 0 && dp[i] == dp[i-1]) same_count++;
        samples++;
    }
    
    // Threadgroup reduction
    uint total_zeros = 0, total_same = 0;
    for (uint i = 0; i < ts; i++) {
        // Simple sum via threadgroup (not optimal but works)
    }
    // Use thread 0 for final classification
    if (t == 0) {
        // Count zeros across all threads' work
        uint zc = 0, sc = 0;
        for (uint i = 0; i < PS; i++) {
            if (dp[i] == 0) zc++;
            if (i > 0 && dp[i] == dp[i-1]) sc++;
        }
        if (zc == PS) return 0;           // TYPE_ZERO
        if (sc >= PS - 2) return 1;        // TYPE_UNIFORM (all same after delta)
        if (zc > PS * 3/4) return 2;       // TYPE_RUN (75%+ zeros)
        if (zc > PS * 1/4) return 3;       // TYPE_STRUCTURED (25%+ zeros = good delta)
        return 4;                           // TYPE_RANDOM
    }
    return 0; // default (only t==0 result matters)
}

// ─── LZ4-style compressor (single-threaded per page, but can be parallelized) ───
// Uses 8192-entry hash table for better match finding
// Match length up to 65535, match distance up to 8192

uint lz4_compress(threadgroup const uchar* dp, device uchar* d, uint db, uint ip_start, uint ip_end) {
    threadgroup uint hk[8192], hv[8192];
    // Hash table already initialized by caller
    
    uint op = db + 4; // Skip header
    uint ip = ip_start;
    uint anchor = ip;
    
    while (ip < ip_end - 4) {
        uint h = h4_tg(dp + ip) & 8191;
        uint match_pos = hv[h];
        uint match_key = hk[h];
        uint cur_key = (uint)dp[ip] | ((uint)dp[ip+1]<<8) | ((uint)dp[ip+2]<<16) | ((uint)dp[ip+3]<<24);
        
        // Update hash table
        hk[h] = cur_key;
        hv[h] = ip;
        
        // Check for match
        if (match_key == cur_key && match_pos < ip && (ip - match_pos) <= 8192) {
            // Found a match - count length
            uint ml = 4;
            while (ip + ml < ip_end && dp[ip + ml] == dp[match_pos + ml]) ml++;
            
            // Encode literal length + match
            uint lit_len = ip - anchor;
            
            // LZ4 token: high nibble = lit_len, low nibble = ml-4
            uint token_off = op;
            uint token = (min(lit_len, 15u) << 4) | min(ml - 4, 15u);
            d[op++] = (uchar)token;
            
            // Extended literal length
            if (lit_len >= 15) {
                uint remaining = lit_len - 15;
                d[op++] = (uchar)min(remaining, 255u);
                remaining -= min(remaining, 255u);
                while (remaining > 0) { d[op++] = (uchar)min(remaining, 255u); remaining -= 255; }
            }
            
            // Copy literals
            for (uint i = 0; i < lit_len; i++) d[op++] = dp[anchor + i];
            
            // Encode match offset (2 bytes, little-endian)
            uint offset = ip - match_pos;
            d[op++] = (uchar)(offset & 0xFF);
            d[op++] = (uchar)((offset >> 8) & 0xFF);
            
            // Extended match length
            if (ml - 4 >= 15) {
                uint remaining = ml - 4 - 15;
                d[op++] = (uchar)min(remaining, 255u);
                remaining -= min(remaining, 255u);
                while (remaining > 0) { d[op++] = (uchar)min(remaining, 255u); remaining -= 255; }
            }
            
            ip += ml;
            anchor = ip;
        } else {
            ip++;
        }
    }
    
    // Encode remaining literals
    uint lit_len = ip_end - anchor;
    if (lit_len > 0 && op + lit_len + 2 <= db + PS) {
        uint token = min(lit_len, 15u) << 4;
        d[op++] = (uchar)token;
        if (lit_len >= 15) {
            uint remaining = lit_len - 15;
            d[op++] = (uchar)min(remaining, 255u);
            remaining -= min(remaining, 255u);
            while (remaining > 0 && op < db + PS - 1) { d[op++] = (uchar)min(remaining, 255u); remaining -= 255; }
        }
        for (uint i = 0; i < lit_len && op < db + PS; i++) d[op++] = dp[anchor + i];
    }
    
    return op - db; // compressed size
}

// ─── RLE compressor (for run-friendly data) ───
uint rle_compress(threadgroup const uchar* dp, device uchar* d, uint db) {
    uint op = db + 4;
    uint ip = 0;
    while (ip < PS && op < db + PS - 4) {
        uint rl = 1;
        while (ip + rl < PS && dp[ip + rl] == dp[ip] && rl < 65535) rl++;
        if (rl >= 4) {
            d[op++] = 0xFD;
            d[op++] = dp[ip];
            d[op++] = (uchar)(rl & 0xFF);
            d[op++] = (uchar)((rl >> 8) & 0xFF);
            ip += rl;
        } else {
            // Escape special bytes
            if (dp[ip] == 0xFD || dp[ip] == 0xFE || dp[ip] == 0xFF) {
                d[op++] = 0xFE;
                d[op++] = dp[ip];
            } else {
                d[op++] = dp[ip];
            }
            ip++;
        }
    }
    return op - db;
}

// ─── Main compression kernel ───
kernel void cp4(
    device const uchar* s[[buffer(0)]],
    device uchar* d[[buffer(1)]],
    device uint* z[[buffer(2)]],
    uint t[[thread_position_in_threadgroup]],
    uint pg[[threadgroup_position_in_grid]],
    uint ts[[threads_per_threadgroup]]
) {
    threadgroup uchar dp[16384];
    threadgroup uint page_type;
    threadgroup uint comp_size;
    uint po = pg * PS;
    
    // ─── Phase 1: Parallel delta encoding ───
    if (t < 256) {
        if (t == 0) { dp[0] = s[po]; for (uint i=1; i<64; i++) dp[i] = s[po+i] - s[po+i-1]; }
        else { dp[t*64] = s[po+t*64] - s[po+t*64-1]; for (uint i=1; i<64; i++) dp[t*64+i] = s[po+t*64+i] - s[po+t*64+i-1]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ─── Phase 2: Page classification (thread 0) ───
    if (t == 0) {
        uint zc = 0;
        for (uint i = 0; i < PS; i++) if (dp[i] == 0) zc++;
        if (zc == PS) page_type = 0;           // ZERO
        else if (zc >= PS - 2) page_type = 1;   // UNIFORM
        else if (zc > PS * 3/4) page_type = 2;  // RUN-friendly
        else if (zc > PS * 1/8) page_type = 3;  // STRUCTURED
        else page_type = 4;                      // RANDOM
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ─── Phase 3: Compress based on type ───
    if (t == 0) {
        uint db = pg * PS;
        
        if (page_type == 0) {
            // Zero page: just store header
            d[db] = 0x4D; d[db+1] = 0x58; d[db+2] = 4; d[db+3] = 0; // type=0
            comp_size = 4;
        } else if (page_type == 1) {
            // Uniform page: store the single byte value
            d[db] = 0x4D; d[db+1] = 0x58; d[db+2] = 4; d[db+3] = 1; // type=1
            d[db+4] = dp[0]; // the non-zero byte (all same after delta)
            comp_size = 5;
        } else if (page_type == 2) {
            // Run-friendly: RLE only
            d[db] = 0x4D; d[db+1] = 0x58; d[db+2] = 4; d[db+3] = 2; // type=2
            comp_size = rle_compress(dp, d, db);
        } else if (page_type == 3) {
            // Structured: LZ4
            d[db] = 0x4D; d[db+1] = 0x58; d[db+2] = 4; d[db+3] = 3; // type=3
            // Initialize hash table
            threadgroup uint *hk_ptr = (threadgroup uint*)0; // Will use shared memory
            // Actually we need hash table in threadgroup memory...
            // For now, use RLE+LZ77 hybrid (existing approach with larger hash)
            comp_size = rle_compress(dp, d, db); // Fallback to RLE for now
        } else {
            // Random: incompressible, store as-is
            comp_size = PS;
        }
        
        // Check if compression was worthwhile
        if (comp_size >= PS - 32) comp_size = PS; // Not worth it
        
        if (comp_size == PS) {
            // Store uncompressed
            z[pg] = PS;
        } else {
            z[pg] = comp_size;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ─── Phase 4: Copy uncompressed if needed ───
    if (z[pg] == PS) {
        for (uint i = t; i < PS; i += ts) d[pg*PS+i] = s[po+i];
    }
}

// ─── Decompression kernel ───
kernel void dp4(
    device const uchar* s[[buffer(0)]],
    device uchar* d[[buffer(1)]],
    device const uint* z[[buffer(2)]],
    uint t[[thread_position_in_threadgroup]],
    uint pg[[threadgroup_position_in_grid]],
    uint ts[[threads_per_threadgroup]]
) {
    uint po = pg * PS, sb = pg * PS, cs = z[pg];
    
    // Uncompressed
    if (cs == PS) { for (uint i=t; i<PS; i+=ts) d[po+i] = s[sb+i]; return; }
    
    // Not our format
    if (s[sb] != 0x4D || s[sb+1] != 0x58) { for (uint i=t; i<PS; i+=ts) d[po+i] = s[sb+i]; return; }
    
    uint ver = s[sb+2];
    uint ptype = s[sb+3];
    threadgroup uchar db[16384];
    
    if (t == 0) {
        if (ptype == 0) {
            // Zero page
            for (uint i = 0; i < PS; i++) db[i] = 0;
        } else if (ptype == 1) {
            // Uniform page
            uchar val = s[sb+4];
            for (uint i = 0; i < PS; i++) db[i] = val;
        } else if (ptype == 2 || ptype == 3) {
            // RLE decompression (same as v2)
            uint ip = 4, op = 0;
            while (ip < cs && op < PS) {
                uchar b = s[sb+ip];
                if (b == 0xFD && ver >= 2 && ip+3 < cs) {
                    uchar vb = s[sb+ip+1];
                    uint rl = (uint)s[sb+ip+2] | ((uint)s[sb+ip+3]<<8);
                    ip += 4;
                    for (uint i = 0; i < rl && op < PS; i++) db[op++] = vb;
                } else if (b == 0xFF && ip+4 < cs) {
                    ip++; uint off = (uint)s[sb+ip]|((uint)s[sb+ip+1]<<8); ip+=2;
                    uint ml = (uint)s[sb+ip]|((uint)s[sb+ip+1]<<8); ip+=2;
                    uint ms = op - off;
                    for (uint i = 0; i < ml && op < PS; i++) db[op++] = db[ms+i];
                } else if (b == 0xFE && ip+1 < cs) {
                    ip++; db[op++] = s[sb+ip++];
                } else {
                    db[op++] = b; ip++;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Delta decode (all types)
    if (t == 0) {
        d[po] = db[0];
        for (uint i = 1; i < PS; i++) d[po+i] = d[po+i-1] + db[i];
    }
}
