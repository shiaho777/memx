import SwiftUI

@main
struct MemXApp: App {
    @StateObject private var appState = AppState()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .frame(minWidth: 720, minHeight: 520)
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified(showsTitle: true))
        .defaultSize(width: 720, height: 520)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About MemX") {
                    NSApplication.shared.orderFrontStandardAboutPanel(
                        options: [.applicationName: "MemX",
                                  .version: "1.0"]
                    )
                }
            }
        }
    }
}

class AppState: ObservableObject {
    @Published var isRunning = false
    @Published var process: Process?
    @Published var outputLog: [LogEntry] = []
    @Published var stats = MemXStats()
    @Published var dylibPath: String = ""
    
    struct LogEntry: Identifiable {
        let id = UUID()
        let text: String
        let time: Date
        let isError: Bool
    }
    
    struct MemXStats {
        var compressions: Int = 0
        var faults: Int = 0
        var bytesSaved: Int64 = 0
        var physicalFootprint: Int64 = 0
    }
    
    init() {
        // Find dylib path - search in order of likelihood
        let searchPaths = dylibSearchPaths()
        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path) {
                dylibPath = path
                return
            }
        }
    }
    
    private func dylibSearchPaths() -> [String] {
        var paths: [String] = []
        // 1. Same directory as executable (inside .app bundle)
        if let exePath = Bundle.main.executablePath {
            let dir = URL(fileURLWithPath: exePath).deletingLastPathComponent().path
            paths.append((dir as NSString).appendingPathComponent("libmemx3.dylib"))
        }
        // 2. Project root (sibling of .app bundle)
        let bundlePath = Bundle.main.bundlePath
        let parent = URL(fileURLWithPath: bundlePath).deletingLastPathComponent().path
        paths.append((parent as NSString).appendingPathComponent("libmemx3.dylib"))
        // 3. Hardcoded fallback (development)
        paths.append("/Users/shiaho/Desktop/memx/libmemx3.dylib")
        return paths
    }
    
    func launch(command: String, arguments: [String] = []) {
        guard !isRunning else { return }
        guard !dylibPath.isEmpty else {
            addLog("Error: libmemx3.dylib not found", isError: true)
            return
        }
        
        outputLog.removeAll()
        stats = MemXStats()
        isRunning = true
        
        let p = Process()
        p.environment = ProcessInfo().environment
        p.environment?["DYLD_INSERT_LIBRARIES"] = dylibPath
        
        // Resolve command path
        let fullCommand: String
        if command.hasPrefix("/") || command.hasPrefix("./") || command.hasPrefix("../") {
            fullCommand = command
        } else {
            fullCommand = "/usr/bin/env"
        }
        p.executableURL = URL(fileURLWithPath: fullCommand)
        
        if fullCommand == "/usr/bin/env" {
            p.arguments = [command] + arguments
        } else {
            p.arguments = arguments
        }
        
        // Capture output
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        
        // Read output asynchronously
        let outputHandle = pipe.fileHandleForReading
        outputHandle.waitForDataInBackgroundAndNotify()
        var observer: NSObjectProtocol?
        observer = NotificationCenter.default.addObserver(
            forName: .NSFileHandleDataAvailable,
            object: outputHandle,
            queue: .main
        ) { [weak self] _ in
            guard let self = self else { return }
            let data = outputHandle.availableData
            if !data.isEmpty {
                if let str = String(data: data, encoding: .utf8) {
                    for line in str.components(separatedBy: "\n") {
                        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !trimmed.isEmpty {
                            self.addLog(trimmed, isError: false)
                            self.parseStats(from: trimmed)
                        }
                    }
                }
                outputHandle.waitForDataInBackgroundAndNotify()
            } else {
                if let obs = observer {
                    NotificationCenter.default.removeObserver(obs)
                }
            }
        }
        
        p.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                self?.isRunning = false
                self?.process = nil
                self?.addLog("Process exited", isError: false)
            }
        }
        
        process = p
        addLog("Launching: \(command) \(arguments.joined(separator: " "))", isError: false)
        addLog("Dylib: \(dylibPath)", isError: false)
        
        do {
            try p.run()
        } catch {
            addLog("Error: \(error.localizedDescription)", isError: true)
            isRunning = false
            process = nil
        }
    }
    
    func stop() {
        process?.interrupt()
        isRunning = false
    }
    
    private func addLog(_ text: String, isError: Bool) {
        outputLog.append(LogEntry(text: text, time: Date(), isError: isError))
    }
    
    private func parseStats(from line: String) {
        // Parse shutdown stats: "Compressed X pages, saved Y MB, resolved Z faults"
        if line.contains("Compressed") && line.contains("pages") {
            let pattern = #"Compressed\s+(\d+)\s+pages.*saved\s+(\d+)\s+MB.*resolved\s+(\d+)\s+faults"#
            if let regex = try? NSRegularExpression(pattern: pattern, options: []),
               let match = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) {
                if let r1 = Range(match.range(at: 1), in: line),
                   let r2 = Range(match.range(at: 2), in: line),
                   let r3 = Range(match.range(at: 3), in: line) {
                    stats.compressions = Int(line[r1]) ?? 0
                    stats.bytesSaved = Int64(line[r2]) ?? 0
                    stats.faults = Int(line[r3]) ?? 0
                }
            }
        }
        // Parse active message: "GPU memory expansion active"
        if line.contains("GPU memory expansion active") {
            addLog("✅ MemX GPU compression active", isError: false)
        }
    }
}
