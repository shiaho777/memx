import SwiftUI

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
    private var monitorTimer: Timer?
    private var lastObservedPIDs: Set<Int32> = []
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
        var processCount = 0
        
        let magic: UInt32 = 0x4D585331
        let statsSize = 160
        
        for file in files {
            guard file.hasPrefix("memx_stats_") else { continue }
            let path = "/tmp/" + file
            guard let data = fm.contents(atPath: path), data.count >= statsSize else { continue }
            
            data.withUnsafeBytes { rawBuf in
                guard let base = rawBuf.baseAddress else { return }
                let ptr = base.assumingMemoryBound(to: UInt32.self)
                guard ptr.pointee == magic else { return }
                
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
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(atPath: "/tmp") else { return }
        
        var procs: [ActiveProcess] = []
        
        for file in files {
            guard file.hasPrefix("memx_stats_") else { continue }
            let pidStr = file.replacingOccurrences(of: "memx_stats_", with: "")
            guard let pid = Int32(pidStr) else { continue }
            
            // Get process name via ps
            let task = Process()
            let pipe = Pipe()
            task.executableURL = URL(fileURLWithPath: "/bin/ps")
            task.arguments = ["-p", "\(pid)", "-o", "comm="]
            task.standardOutput = pipe
            try? task.run()
            task.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let name = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "PID \(pid)"
            
            // Get RSS
            let task2 = Process()
            let pipe2 = Pipe()
            task2.executableURL = URL(fileURLWithPath: "/bin/ps")
            task2.arguments = ["-p", "\(pid)", "-o", "rss="]
            task2.standardOutput = pipe2
            try? task2.run()
            task2.waitUntilExit()
            let data2 = pipe2.fileHandleForReading.readDataToEndOfFile()
            let rssStr = String(data: data2, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "0"
            let rssKB = Int64(rssStr) ?? 0
            let rssMB = rssKB / 1024
            
            procs.append(ActiveProcess(pid: pid, name: name, memoryMB: rssMB, memxActive: true))
        }
        
        activeProcesses = procs.sorted { lhs, rhs in
            if lhs.memoryMB == rhs.memoryMB { return lhs.pid < rhs.pid }
            return lhs.memoryMB > rhs.memoryMB
        }
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
}
