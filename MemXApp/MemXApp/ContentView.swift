import SwiftUI
import AppKit

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        VStack(spacing: 0) {
            headerBar
            Divider()
            HStack(spacing: 0) {
                leftPanel.frame(maxWidth: .infinity)
                Divider()
                rightPanel.frame(maxWidth: .infinity)
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .frame(minWidth: 780, minHeight: 520)
    }
    
    // MARK: - Header
    
    private var headerBar: some View {
        HStack(spacing: 16) {
            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .fill(LinearGradient(colors: [.blue, .purple], startPoint: .topLeading, endPoint: .bottomTrailing))
                    .frame(width: 40, height: 40)
                Image(systemName: "memorychip.fill")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundColor(.white)
            }
            
            VStack(alignment: .leading, spacing: 1) {
                Text("MemX")
                    .font(.system(size: 16, weight: .bold))
                Text("Runtime Monitor")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // System RAM
            if appState.systemMemory.total > 0 {
                HStack(spacing: 8) {
                    Image(systemName: "internaldrive")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                    VStack(alignment: .trailing, spacing: 1) {
                        Text(String(format: "%.1f / %.0f GB", appState.systemMemory.usedGB, appState.systemMemory.totalGB))
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                        Text("System RAM")
                            .font(.system(size: 8))
                            .foregroundColor(.secondary)
                    }
                    GeometryReader { geo in
                        let pct = appState.systemMemory.usagePercent / 100.0
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 3).fill(Color.gray.opacity(0.2))
                            RoundedRectangle(cornerRadius: 3)
                                .fill(pct > 0.8 ? Color.red : pct > 0.6 ? Color.orange : Color.green)
                                .frame(width: geo.size.width * min(pct, 1.0))
                        }
                    }
                    .frame(width: 80, height: 6)
                }
            }
            
            Divider().frame(height: 28)
            
            VStack(alignment: .trailing, spacing: 6) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(appState.isActive ? Color.green : Color.secondary.opacity(0.45))
                        .frame(width: 8, height: 8)
                    Text(appState.isActive ? "Workloads Live" : "Passive Monitor")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(appState.isActive ? .green : .secondary)
                }
                HStack(spacing: 10) {
                    Toggle("Auto", isOn: $appState.autoRefresh)
                        .toggleStyle(.switch)
                        .controlSize(.small)
                        .font(.system(size: 10))
                    Button(action: appState.refreshNow) {
                        Label("Refresh", systemImage: "arrow.clockwise")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                    .help("Refresh runtime snapshot")
                }
                if let lastRefreshAt = appState.lastRefreshAt {
                    Text(lastRefreshAt, style: .time)
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color(nsColor: .controlBackgroundColor))
    }
    
    // MARK: - Left Panel
    
    private var leftPanel: some View {
        ScrollView {
            VStack(spacing: 20) {
                expansionGauge
                Divider().padding(.horizontal, 20)
                statsSection
                Divider().padding(.horizontal, 20)
                processSection
                Divider().padding(.horizontal, 20)
                storeServiceSection
                Divider().padding(.horizontal, 20)
                capsuleSection
                Divider().padding(.horizontal, 20)
                infoSection
            }
            .padding(20)
        }
    }

    // MARK: - Store Service

    private var storeServiceSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Store Service")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.secondary)
                Spacer()
                HStack(spacing: 4) {
                    Circle()
                        .fill(appState.storeRunning ? Color.green : Color.gray.opacity(0.5))
                        .frame(width: 7, height: 7)
                    Text(appState.storeRunning ? "running" : "stopped")
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundColor(appState.storeRunning ? .green : .secondary)
                }
            }

            HStack(spacing: 8) {
                Button {
                    appState.storeStart()
                } label: {
                    Label("Start", systemImage: "play.fill")
                        .font(.system(size: 10, weight: .medium))
                }
                .buttonStyle(.bordered)
                .disabled(appState.storeRunning)

                Button {
                    appState.storeStop()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                        .font(.system(size: 10, weight: .medium))
                }
                .buttonStyle(.bordered)
                .disabled(!appState.storeRunning)
            }

            if appState.storeStats.isEmpty {
                Text("Unprivileged capsule store · PUT/GET/LIST/COMMIT/VERIFY over UDS")
                    .font(.system(size: 9))
                    .foregroundColor(Color(nsColor: .tertiaryLabelColor))
            } else {
                Text(appState.storeStats)
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
            }
        }
    }

    // MARK: - Capsule Browser

    private var capsuleSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Capsule Store")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.secondary)
                Spacer()
                if appState.capsule.valid {
                    Text("v\(appState.capsule.version)")
                        .font(.system(size: 10, weight: .medium, design: .rounded))
                        .foregroundColor(.secondary)
                }
            }

            HStack(spacing: 8) {
                Button {
                    let panel = NSOpenPanel()
                    panel.canChooseDirectories = true
                    panel.canChooseFiles = false
                    panel.message = "Select a capsule directory (spill.bin / ledger.bin)"
                    if panel.runModal() == .OK, let url = panel.url {
                        appState.loadCapsule(atPath: url.path)
                    }
                } label: {
                    Label("Browse…", systemImage: "folder")
                        .font(.system(size: 10, weight: .medium))
                }
                .buttonStyle(.bordered)

                if appState.capsule.valid {
                    Button {
                        appState.verifyCapsule()
                    } label: {
                        Label("Verify", systemImage: "checkmark.shield")
                            .font(.system(size: 10, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                    Text(appState.capsule.verifyStatus)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(appState.capsule.verifyStatus == "ok" ? .green :
                                         appState.capsule.verifyStatus == "corrupt" ? .red : .secondary)
                }
            }

            if appState.capsule.path.isEmpty {
                Text("No capsule selected")
                    .font(.system(size: 10))
                    .foregroundColor(Color(nsColor: .tertiaryLabelColor))
            } else if !appState.capsule.valid {
                Text("Not a valid capsule: \(appState.capsule.verifyDetail)")
                    .font(.system(size: 10))
                    .foregroundColor(.red)
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    Text(URL(fileURLWithPath: appState.capsule.path).lastPathComponent)
                        .font(.system(size: 11, weight: .medium))
                    Text(String(format: "%lld pages · %.1f MB logical · %.1f MB spill",
                                appState.capsule.entCount,
                                Double(appState.capsule.pageBytes) / 1048576.0,
                                Double(appState.capsule.spillBytes) / 1048576.0))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.secondary)
                    if !appState.capsule.verifyDetail.isEmpty {
                        Text(appState.capsule.verifyDetail)
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundColor(.orange)
                    }
                }

                if appState.capsule.hasManifest {
                    ForEach(appState.capsule.segments) { seg in
                        HStack {
                            Image(systemName: "square.stack.3d.up")
                                .font(.system(size: 9))
                                .foregroundColor(.secondary)
                            Text(seg.name)
                                .font(.system(size: 10, design: .monospaced))
                            Spacer()
                            Text(String(format: "%lldp · %.1f MB", seg.pages, Double(seg.nbytes) / 1048576.0))
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                        .padding(4)
                        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                    }
                } else {
                    Text("No manifest (unnamed capsule)")
                        .font(.system(size: 9))
                        .foregroundColor(Color(nsColor: .tertiaryLabelColor))
                }
            }
        }
    }
    
    // MARK: - Expansion Gauge
    
    private var expansionGauge: some View {
        VStack(spacing: 10) {
            Text("Managed Memory Expansion")
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(.secondary)
            
            ZStack {
                Circle()
                    .stroke(Color.gray.opacity(0.12), lineWidth: 12)
                    .frame(width: 160, height: 160)
                
                Circle()
                    .trim(from: 0, to: min(CGFloat(appState.stats.expansionRatio) / 100.0, 1.0))
                    .stroke(LinearGradient(colors: [.blue, .purple, .pink], startPoint: .topLeading, endPoint: .bottomTrailing),
                            style: StrokeStyle(lineWidth: 12, lineCap: .round))
                    .frame(width: 160, height: 160)
                    .rotationEffect(.degrees(-90))
                    .animation(.easeOut(duration: 0.5), value: appState.stats.expansionRatio)
                
                VStack(spacing: 2) {
                    Text(appState.stats.expansionRatio > 0 ? String(format: "%.0f×", appState.stats.expansionRatio) : "—")
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                    Text("expansion")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary)
                }
            }
            
            HStack(spacing: 24) {
                MiniStat(label: "Virtual", value: appState.stats.virtualMB > 0 ? "\(appState.stats.virtualMB / 1024) GB" : "—")
                MiniStat(label: "Physical", value: appState.stats.physicalMB > 0 ? "\(appState.stats.physicalMB) MB" : "—")
                MiniStat(label: "Saved", value: appState.stats.bytesSaved > 0 ? "\(appState.stats.bytesSaved / (1024 * 1024)) MB" : "—")
            }
        }
    }
    
    // MARK: - Stats Section
    
    private var statsSection: some View {
        VStack(spacing: 10) {
            Text("Compression Stats")
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                StatCard(icon: "arrow.down.doc.fill", iconColor: .blue,
                         title: "Compressed", value: fmt(appState.stats.compressions), unit: "pages")
                StatCard(icon: "bolt.fill", iconColor: .orange,
                         title: "Faults", value: fmt(appState.stats.faults), unit: "resolved")
                StatCard(icon: "doc.on.doc.fill", iconColor: .purple,
                         title: "Dedup Hits", value: fmt(appState.stats.dedupHits), unit: "")
                StatCard(icon: "arrow.forward.fill", iconColor: .green,
                         title: "Prefetch", value: fmt(appState.stats.prefetchCount), unit: "")
            }
        }
    }
    
    // MARK: - Active Processes
    
    private var processSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Active Managed Workloads")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.secondary)
                Spacer()
                if appState.stats.processCount > 0 {
                    Text("\(appState.stats.processCount) active")
                        .font(.system(size: 10, weight: .medium, design: .rounded))
                        .foregroundColor(.green)
                }
            }
            
            if appState.activeProcesses.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "terminal")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary)
                    Text("No active MemX-managed workloads yet")
                        .font(.system(size: 10))
                        .foregroundColor(Color(nsColor: .tertiaryLabelColor))
                }
            } else {
                ForEach(appState.activeProcesses) { proc in
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 11))
                            .foregroundColor(.green)
                        
                        VStack(alignment: .leading, spacing: 1) {
                            Text(proc.name)
                                .font(.system(size: 11, weight: .medium))
                            Text("PID \(proc.pid)")
                                .font(.system(size: 9, design: .monospaced))
                                .foregroundColor(Color(nsColor: .tertiaryLabelColor))
                        }
                        
                        Spacer()
                        
                        VStack(alignment: .trailing, spacing: 1) {
                            Text("\(proc.memoryMB) MB")
                                .font(.system(size: 10, weight: .medium, design: .monospaced))
                                .foregroundColor(.secondary)
                            Text(proc.memxActive ? "managed" : "observed")
                                .font(.system(size: 8))
                                .foregroundColor(Color(nsColor: .tertiaryLabelColor))
                        }
                    }
                    .padding(6)
                    .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
                    .clipShape(RoundedRectangle(cornerRadius: 5))
                }
            }
        }
    }
    
    // MARK: - Info Section
    
    private var infoSection: some View {
        VStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Product Direction")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.secondary)
                
                InfoRow(icon: "memorychip", text: "Host apps embed MemX directly")
                InfoRow(icon: "dial.high", text: "Contexts add quotas, ownership tracking, and telemetry")
                InfoRow(icon: "gpu", text: "Cold pages are compressed with Metal")
                InfoRow(icon: "square.stack.3d.down.right", text: "Managed buffers stay inside the host process")
            }
            
            VStack(alignment: .leading, spacing: 6) {
                Text("Target Workloads")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.secondary)
                
                InfoRow(icon: "cpu", text: "Local AI runtimes and tensor caches")
                InfoRow(icon: "server.rack", text: "Vector DB, cache, and search engines")
                InfoRow(icon: "hammer", text: "Heavy desktop and developer tools")
                InfoRow(icon: "gauge", text: "Quota-aware memory tiers inside one app")
            }
            
        }
    }
    
    // MARK: - Right Panel (Log)
    
    private var rightPanel: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Activity Log")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.secondary)
                Spacer()
                if !appState.outputLog.isEmpty {
                    Button("Clear") { appState.outputLog.removeAll() }
                        .font(.system(size: 10))
                        .buttonStyle(.borderless)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
            
            Divider()
            
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 1) {
                        if appState.outputLog.isEmpty {
                            VStack(spacing: 12) {
                                Image(systemName: "terminal")
                                    .font(.system(size: 28))
                                    .foregroundColor(Color(nsColor: .quaternaryLabelColor))
                                Text("Start a MemX-managed workload to see runtime events")
                                    .font(.system(size: 12))
                                    .foregroundColor(Color(nsColor: .tertiaryLabelColor))
                            }
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .padding(.top, 60)
                        }
                        ForEach(appState.outputLog) { entry in
                            logRow(entry).id(entry.id)
                        }
                    }
                    .padding(8)
                }
                .onChange(of: appState.outputLog.count) {
                    if let last = appState.outputLog.last {
                        withAnimation(.easeOut(duration: 0.15)) {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }
        }
    }
    
    // MARK: - Helpers
    
    @ViewBuilder
    private func logRow(_ entry: AppState.LogEntry) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(entry.time, style: .time)
                .font(.system(size: 9, design: .monospaced))
                .foregroundColor(Color(nsColor: .quaternaryLabelColor))
                .frame(width: 52, alignment: .leading)
            switch entry.category {
            case .success: Image(systemName: "checkmark.circle.fill").font(.system(size: 10)).foregroundColor(.green)
            case .error:   Image(systemName: "xmark.circle.fill").font(.system(size: 10)).foregroundColor(.red)
            case .warning: Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 10)).foregroundColor(.orange)
            case .data:    Image(systemName: "chart.bar.fill").font(.system(size: 10)).foregroundColor(.blue)
            case .info:    Image(systemName: "info.circle.fill").font(.system(size: 10)).foregroundColor(.secondary)
            }
            Text(entry.text)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(entry.isError ? .red : .primary)
        }
        .padding(.vertical, 1).padding(.horizontal, 4)
        .background(entry.isError ? Color.red.opacity(0.05) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 3))
    }
    
    private func fmt(_ n: Int64) -> String {
        if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.1fK", Double(n) / 1_000) }
        return "\(n)"
    }
}

// MARK: - Subviews

struct MiniStat: View {
    let label: String
    let value: String
    var body: some View {
        VStack(spacing: 2) {
            Text(value).font(.system(size: 12, weight: .semibold, design: .rounded))
            Text(label).font(.system(size: 8)).foregroundColor(.secondary)
        }
    }
}

struct StatCard: View {
    let icon: String; let iconColor: Color; let title: String; let value: String; let unit: String
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: icon).font(.system(size: 10)).foregroundColor(iconColor)
                Text(title).font(.system(size: 9)).foregroundColor(.secondary)
            }
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value).font(.system(size: 16, weight: .bold, design: .rounded))
                if !unit.isEmpty { Text(unit).font(.system(size: 9)).foregroundColor(.secondary) }
            }
        }
        .padding(8)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

struct InfoRow: View {
    let icon: String; let text: String
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon).font(.system(size: 11)).foregroundColor(.accentColor).frame(width: 16)
            Text(text).font(.system(size: 11)).foregroundColor(.secondary)
        }
    }
}

#Preview {
    ContentView().environmentObject(AppState()).frame(width: 780, height: 520)
}
