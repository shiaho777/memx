import SwiftUI
import Darwin

struct CapsuleSegment: Identifiable {
    let id = UUID()
    let name: String
    let pages: Int64
    let nbytes: Int64
}

struct CapsuleInfo {
    var path: String = ""
    var valid = false
    var version: UInt32 = 0
    var entCount: UInt64 = 0
    var spillBytes: UInt64 = 0
    var pageBytes: UInt64 = 0
    var segments: [CapsuleSegment] = []
    var hasManifest = false
    var verifyStatus: String = "not verified"
    var verifyDetail: String = ""
}

@main
struct MemXApp: App {
    @StateObject private var appState = AppState()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .frame(minWidth: 800, minHeight: 600)
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified(showsTitle: true))
        .defaultSize(width: 900, height: 640)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About MemX") {
                    NSApplication.shared.orderFrontStandardAboutPanel(
                        options: [.applicationName: "MemX",
                                  .version: "3.0"]
                    )
                }
            }
        }
    }
}

class AppState: ObservableObject {
    @Published var autoRefresh = true
    @Published var isActive = false
    @Published var stats = MemXStats()
    @Published var systemMemory = SystemMemory()
    @Published var lastRefreshAt: Date?
    @Published var outputLog: [LogEntry] = []
    @Published var activeProcesses: [ActiveProcess] = []
    @Published var capsule = CapsuleInfo()
    @Published var storeRunning = false
    @Published var storeStats = ""
    private var monitorTimer: Timer?
    private var storedProcess: Process?
    private var lastObservedPIDs: Set<Int32> = []
    private var lastLivePids: Set<Int32> = []
    private var didLogInitialState = false
    
    // MARK: - Types
    
    struct LogEntry: Identifiable {
        let id = UUID()
        let text: String
        let time: Date
        let isError: Bool
        let category: LogCategory
        enum LogCategory { case info, success, warning, error, data }
    }
    
    struct MemXStats {
        var compressions: Int64 = 0
        var faults: Int64 = 0
        var bytesSaved: Int64 = 0
        var dedupHits: Int64 = 0
        var prefetchCount: Int64 = 0
        var prefetchHits: Int64 = 0
        var virtualMB: Int64 = 0
        var physicalMB: Int64 = 0
        var expansionRatio: Double = 0
        var integrityOK: Bool = true
        var processCount: Int = 0
        var compressedPages: Int64 = 0
        var hotPages: Int64 = 0
        var spillMB: Int64 = 0
        var liveContexts: Int64 = 0
    }
    
    struct ActiveProcess: Identifiable {
        let id = UUID()
        let pid: Int32
        let name: String
        let memoryMB: Int64
        let memxActive: Bool
    }
    
    struct SystemMemory {
        var total: Int64 = 0
        var used: Int64 = 0
        var free: Int64 = 0
        var usedGB: Double { Double(used) / 1_073_741_824 }
        var totalGB: Double { Double(total) / 1_073_741_824 }
        var usagePercent: Double { total > 0 ? Double(used) / Double(total) * 100 : 0 }
    }
    
    // MARK: - Init
    
    init() {
        refreshSnapshot(trigger: "startup")
        startGlobalMonitor()
    }
    
    deinit {
        monitorTimer?.invalidate()
    }
    
    // MARK: - Monitor
    
    private func startGlobalMonitor() {
        monitorTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let self, self.autoRefresh else { return }
            self.refreshSnapshot(trigger: "auto")
        }
    }
    
    func refreshNow() {
        refreshSnapshot(trigger: "manual")
    }
    
    func refreshSnapshot(trigger: String) {
        let total = ProcessInfo().physicalMemory
        var vmStats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &vmStats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        if result == KERN_SUCCESS {
            let pageSize = vm_kernel_page_size
            let used = Int64(total) - Int64(vmStats.free_count + vmStats.inactive_count) * Int64(pageSize)
            systemMemory = SystemMemory(total: Int64(total), used: max(0, used), free: Int64(vmStats.free_count) * Int64(pageSize))
        }
        scanGlobalStats()
        scanActiveProcesses()
        storeProbe()
        lastRefreshAt = Date()
        recordMonitorActivity(trigger: trigger)
    }
    
    // MARK: - Global Stats from Shared Memory
    
    private func scanGlobalStats() {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(atPath: "/tmp") else { return }

        var totalCompressions: Int64 = 0
        var totalFaults: Int64 = 0
        var totalBytesSaved: Int64 = 0
        var totalDedupHits: Int64 = 0
        var totalPrefetchCount: Int64 = 0
        var totalPrefetchHits: Int64 = 0
        var totalVirtualMB: Int64 = 0
        var totalPoolUsed: Int64 = 0
        var totalPagesResident: Int64 = 0
        var totalCompressedPages: Int64 = 0
        var totalHotPages: Int64 = 0
        var totalSpillBytes: Int64 = 0
        var totalLiveContexts: Int64 = 0
        var livePids = Set<Int32>()
        var processCount = 0

        let magicV2: UInt32 = 0x4D585332
        let magicV1: UInt32 = 0x4D585331
        let statsSizeV2 = 176
        let statsSizeV1 = 160

        for file in files {
            guard file.hasPrefix("memx_stats_") else { continue }
            let path = "/tmp/" + file
            guard let data = fm.contents(atPath: path), data.count >= 8 else { continue }

            data.withUnsafeBytes { rawBuf in
                guard let base = rawBuf.baseAddress else { return }
                let ptr = base.assumingMemoryBound(to: UInt32.self)
                let magic = ptr.pointee
                guard magic == magicV2 || magic == magicV1 else { return }
                // v2: magic u32 | ver u16 | pad u16 | pid u32 | pad u32 | u64 fields
                // v1: magic u32 | pid u32 | u64 fields
                let pid = Int32(bitPattern: magic == magicV2 ? ptr[2] : ptr[1])
                guard kill(pid, 0) == 0 || errno == EPERM else { return }
                livePids.insert(pid)

                if magic == magicV2 {
                    guard data.count >= statsSizeV2 else { return }
                    // layout: magic u32 | ver u16 + pad u16 | pid u32 + pad u32 | u64 fields from offset 8
                    let ver = ptr[1] & 0xFFFF
                    guard ver == 2 else { return }
                    let fieldBase = base.assumingMemoryBound(to: UInt64.self) + 1
                    totalCompressions += Int64(fieldBase[3])
                    totalFaults += Int64(fieldBase[4])
                    totalBytesSaved += Int64(fieldBase[5])
                    totalDedupHits += Int64(fieldBase[6])
                    totalPrefetchCount += Int64(fieldBase[7])
                    totalPrefetchHits += Int64(fieldBase[8])
                    totalVirtualMB += Int64(fieldBase[9])
                    totalPoolUsed += Int64(fieldBase[10])
                    totalCompressedPages += Int64(fieldBase[12])
                    totalPagesResident += Int64(fieldBase[13])
                    totalHotPages += Int64(fieldBase[14])
                    totalSpillBytes += Int64(fieldBase[15])
                    totalLiveContexts += Int64(fieldBase[16])
                } else {
                    guard data.count >= statsSizeV1 else { return }
                    let fieldBase = base.assumingMemoryBound(to: UInt64.self) + 1
                    totalCompressions += Int64(fieldBase[0])
                    totalFaults += Int64(fieldBase[1])
                    totalBytesSaved += Int64(fieldBase[2])
                    totalDedupHits += Int64(fieldBase[3])
                    totalPrefetchCount += Int64(fieldBase[4])
                    totalPrefetchHits += Int64(fieldBase[5])
                    totalVirtualMB += Int64(fieldBase[6])
                    totalPoolUsed += Int64(fieldBase[7])
                    totalPagesResident += Int64(fieldBase[10])
                }
                processCount += 1
            }
        }

        stats.compressions = totalCompressions
        stats.faults = totalFaults
        stats.bytesSaved = totalBytesSaved
        stats.dedupHits = totalDedupHits
        stats.prefetchCount = totalPrefetchCount
        stats.prefetchHits = totalPrefetchHits
        stats.virtualMB = totalVirtualMB
        stats.processCount = processCount
        stats.compressedPages = totalCompressedPages
        stats.hotPages = totalHotPages
        stats.spillMB = totalSpillBytes / (1024 * 1024)
        stats.liveContexts = totalLiveContexts
        lastLivePids = livePids

        if processCount > 0 {
            let physicalBytes = totalPagesResident * 16384 + totalPoolUsed
            stats.physicalMB = Int64(physicalBytes / (1024 * 1024))
            let physicalGB = Double(stats.physicalMB) / 1024.0
            stats.expansionRatio = physicalGB > 0.01 ? Double(totalVirtualMB) / 1024.0 / physicalGB : 0
            isActive = true
        } else {
            stats.physicalMB = 0
            stats.expansionRatio = 0
            isActive = false
        }
    }
    
    // MARK: - Active Process List
    
    private func scanActiveProcesses() {
        guard !lastLivePids.isEmpty else {
            activeProcesses = []
            return
        }
        let pidList = lastLivePids.map(String.init).sorted().joined(separator: ",")

        let task = Process()
        let pipe = Pipe()
        task.executableURL = URL(fileURLWithPath: "/bin/ps")
        task.arguments = ["-p", pidList, "-o", "pid=,comm=,rss="]
        task.standardOutput = pipe
        do { try task.run() } catch { return }
        task.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard let text = String(data: data, encoding: .utf8) else { return }

        var procs: [ActiveProcess] = []
        for line in text.split(separator: "\n") {
            let fields = line.split(separator: " ", omittingEmptySubsequences: true)
            guard fields.count >= 3, let pid = Int32(fields[0]) else { continue }
            let rssKB = Int64(fields[fields.count - 1]) ?? 0
            let name = fields[1..<(fields.count - 1)].joined(separator: " ")
            procs.append(ActiveProcess(pid: pid, name: name, memoryMB: rssKB / 1024, memxActive: true))
        }

        activeProcesses = procs.sorted { lhs, rhs in
            if lhs.memoryMB == rhs.memoryMB { return lhs.pid < rhs.pid }
            return lhs.memoryMB > rhs.memoryMB
        }
    }
    
    // MARK: - Capsule Browser

    func loadCapsule(atPath path: String) {
        var info = CapsuleInfo()
        info.path = path
        let fm = FileManager.default

        guard let led = fm.contents(atPath: path + "/ledger.bin"), led.count >= 72 else {
            info.verifyDetail = "ledger.bin missing or truncated"
            capsule = info
            return
        }
        led.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            let u32 = base.assumingMemoryBound(to: UInt32.self)
            guard u32[0] == 0x4D584350 else { return }
            info.version = u32[1]
            let u64 = base.assumingMemoryBound(to: UInt64.self)
            info.entCount = u64[1]
            info.spillBytes = u64[2]
            info.pageBytes = u64[3]
            info.valid = true
        }

        if let man = fm.contents(atPath: path + "/manifest.bin"), man.count >= 16 {
            man.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return }
                let u32 = base.assumingMemoryBound(to: UInt32.self)
                guard u32[0] == 0x4D585347, u32[1] == 1 else { return }
                let count = Int(u32[2])
                var segs: [CapsuleSegment] = []
                let entrySize = 88
                for i in 0..<min(count, 128) {
                    let off = 16 + i * entrySize
                    guard man.count >= off + entrySize else { break }
                    man.withUnsafeBytes { r2 in
                        guard let b2 = r2.baseAddress else { return }
                        let nameEnd = off + 64
                        var nameBytes: [UInt8] = []
                        for j in off..<nameEnd {
                            let c = man[man.index(man.startIndex, offsetBy: j)]
                            if c == 0 { break }
                            nameBytes.append(c)
                        }
                        guard let name = String(bytes: nameBytes, encoding: .utf8), !name.isEmpty else { return }
                        let b32 = (b2 + nameEnd).assumingMemoryBound(to: UInt32.self)
                        let b64 = (b2 + nameEnd).assumingMemoryBound(to: UInt64.self)
                        let pages = UInt64(b32[1])
                        let nbytes = b64[2]
                        segs.append(CapsuleSegment(name: name, pages: Int64(pages), nbytes: Int64(nbytes)))
                    }
                }
                info.segments = segs
                info.hasManifest = true
            }
        }

        info.verifyStatus = "not verified"
        info.verifyDetail = ""
        capsule = info
    }

    func verifyCapsule() {
        guard capsule.valid else { return }
        let vesselPath = Bundle.main.bundleURL
            .appendingPathComponent("Contents/MacOS/memx_capsule_vessel").path
        guard FileManager.default.fileExists(atPath: vesselPath) else {
            capsule.verifyStatus = "unavailable"
            capsule.verifyDetail = "vessel binary not bundled"
            return
        }
        let task = Process()
        let pipe = Pipe()
        task.executableURL = URL(fileURLWithPath: vesselPath)
        task.arguments = ["--dir", capsule.path, "--verify"]
        task.standardOutput = pipe
        task.standardError = Pipe()
        do { try task.run() } catch {
            capsule.verifyStatus = "error"
            capsule.verifyDetail = error.localizedDescription
            return
        }
        task.waitUntilExit()
        let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        var status = "unknown"
        var bad = ""
        var pages = ""
        for line in out.split(separator: "\n") {
            if line.hasPrefix("VESSEL_VERIFY_STATUS=") {
                status = String(line.dropFirst("VESSEL_VERIFY_STATUS=".count))
            } else if line.hasPrefix("VESSEL_VERIFY_BAD=") {
                bad = String(line.dropFirst("VESSEL_VERIFY_BAD=".count))
            } else if line.hasPrefix("VESSEL_VERIFY_PAGES=") {
                pages = String(line.dropFirst("VESSEL_VERIFY_PAGES=".count))
            }
        }
        capsule.verifyStatus = status.lowercased()
        capsule.verifyDetail = pages.isEmpty ? "" : "\(bad)/\(pages) pages failed CRC"
    }

    // MARK: - Store Service

    private var storeSockPath: String {
        let root = NSHomeDirectory() + "/Library/Application Support/MemX/store"
        return root + "/store.sock"
    }

    private var bundledStoredPath: String {
        Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/memx_stored").path
    }

    func storeStart() {
        guard storedProcess == nil else { return }
        let root = NSHomeDirectory() + "/Library/Application Support/MemX/store"
        try? FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: bundledStoredPath)
        p.arguments = ["--root", root]
        p.standardError = Pipe()
        do {
            try p.run()
            storedProcess = p
            storeRunning = true
            log("store service started pid=\(p.processIdentifier)", category: .success)
        } catch {
            log("store start failed: \(error.localizedDescription)", category: .error)
        }
    }

    func storeStop() {
        if let p = storedProcess, p.isRunning {
            p.terminate()
            storedProcess = nil
            storeRunning = false
            log("store service stopped", category: .info)
        }
    }

    func storeProbe() {
        guard FileManager.default.fileExists(atPath: storeSockPath) else {
            storeRunning = storedProcess?.isRunning ?? false
            storeStats = ""
            return
        }
        storeRunning = true
        let sockFd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard sockFd >= 0 else { return }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = Array(storeSockPath.utf8)
        withUnsafeMutableBytes(of: &addr.sun_path) { dst in
            dst.copyBytes(from: pathBytes)
        }
        let connected = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(sockFd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard connected == 0 else {
            close(sockFd)
            return
        }
        let cmd: [UInt8] = Array("STAT".utf8)
        var beLen = UInt32(cmd.count).bigEndian
        var payload = Data()
        withUnsafeBytes(of: &beLen) { payload.append(contentsOf: $0) }
        payload.append(contentsOf: cmd)
        _ = payload.withUnsafeBytes { send(sockFd, $0.baseAddress, $0.count, 0) }
        var lenBuf = [UInt8](repeating: 0, count: 4)
        var got = 0
        while got < 4 {
            let n = lenBuf.withUnsafeMutableBytes { ptr in
                recv(sockFd, ptr.baseAddress! + got, 4 - got, 0)
            }
            if n <= 0 { break }
            got += n
        }
        if got == 4 {
            let rlen = Int(lenBuf[0]) << 24 | Int(lenBuf[1]) << 16 | Int(lenBuf[2]) << 8 | Int(lenBuf[3])
            if rlen > 0 && rlen < 4096 {
                var resp = [UInt8](repeating: 0, count: rlen)
                var got2 = 0
                while got2 < rlen {
                    let n = resp.withUnsafeMutableBytes { ptr in
                        recv(sockFd, ptr.baseAddress! + got2, rlen - got2, 0)
                    }
                    if n <= 0 { break }
                    got2 += n
                }
                if let text = String(bytes: resp.prefix(got2), encoding: .utf8) {
                    storeStats = text.replacingOccurrences(of: "OK stats ", with: "")
                }
            }
        }
        // send QUIT to close cleanly
        let q: [UInt8] = Array("QUIT".utf8)
        var qbe = UInt32(q.count).bigEndian
        var qpayload = Data()
        withUnsafeBytes(of: &qbe) { qpayload.append(contentsOf: $0) }
        qpayload.append(contentsOf: q)
        _ = qpayload.withUnsafeBytes { send(sockFd, $0.baseAddress, $0.count, 0) }
        var sink = [UInt8](repeating: 0, count: 64)
        _ = recv(sockFd, &sink, 64, 0)
        close(sockFd)
    }

    // MARK: - Logging    
    private func recordMonitorActivity(trigger: String) {
        let currentPIDs = Set(activeProcesses.map(\.pid))
        if !didLogInitialState {
            didLogInitialState = true
            addLog("monitor ready: trigger=\(trigger) active_workloads=\(activeProcesses.count)", isError: false, category: .info)
        }
        
        let newPIDs = currentPIDs.subtracting(lastObservedPIDs)
        let removedPIDs = lastObservedPIDs.subtracting(currentPIDs)
        
        for pid in newPIDs.sorted() {
            if let proc = activeProcesses.first(where: { $0.pid == pid }) {
                addLog("attached pid=\(proc.pid) name=\(proc.name) rss_mb=\(proc.memoryMB)", isError: false, category: .success)
            }
        }
        for pid in removedPIDs.sorted() {
            addLog("detached pid=\(pid)", isError: false, category: .warning)
        }
        
        if !newPIDs.isEmpty || !removedPIDs.isEmpty || trigger == "manual" {
            addLog("snapshot workloads=\(stats.processCount) virtual_mb=\(stats.virtualMB) physical_mb=\(stats.physicalMB) saved_mb=\(stats.bytesSaved) faults=\(stats.faults)", isError: false, category: .data)
        }
        
        lastObservedPIDs = currentPIDs
    }
    
    private func addLog(_ text: String, isError: Bool, category: LogEntry.LogCategory) {
        outputLog.append(LogEntry(text: text, time: Date(), isError: isError, category: category))
    }

    private func log(_ text: String, category: LogEntry.LogCategory) {
        addLog(text, isError: category == .error, category: category)
    }
}
