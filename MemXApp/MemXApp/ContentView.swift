import SwiftUI

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    @State private var commandText: String = ""
    @State private var argsText: String = ""
    @State private var selectedPreset: Int = 0
    @State private var showFilePicker = false
    
    private let presets = [
        (name: "Custom", cmd: "", args: ""),
        (name: "Python 3", cmd: "python3", args: ""),
        (name: "Python Script", cmd: "python3", args: "script.py"),
        (name: "Shell", cmd: "/bin/bash", args: ""),
        (name: "Test: 1GB", cmd: "/Users/shiaho/Desktop/memx/test_1gb", args: ""),
        (name: "Test: Physical", cmd: "/Users/shiaho/Desktop/memx/test_phys", args: ""),
        (name: "Test: Real Workload", cmd: "/Users/shiaho/Desktop/memx/test_realworkload", args: ""),
    ]
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            headerBar
            
            Divider()
            
            // Main content
            HSplitView {
                // Left: Launcher
                launcherPanel
                    .frame(minWidth: 280, maxWidth: 360)
                
                // Right: Output + Stats
                VStack(spacing: 0) {
                    statsBar
                    Divider()
                    outputPanel
                }
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
    
    // MARK: - Header
    
    private var headerBar: some View {
        HStack(spacing: 12) {
            // Icon
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(
                        LinearGradient(colors: [.blue.opacity(0.8), .purple.opacity(0.8)],
                                       startPoint: .topLeading, endPoint: .bottomTrailing)
                    )
                    .frame(width: 36, height: 36)
                Text("M")
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
            }
            
            VStack(alignment: .leading, spacing: 2) {
                Text("MemX")
                    .font(.headline)
                Text("GPU Memory Expansion")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // Status indicator
            HStack(spacing: 6) {
                Circle()
                    .fill(appState.isRunning ? Color.green : Color.gray.opacity(0.5))
                    .frame(width: 8, height: 8)
                Text(appState.isRunning ? "Running" : "Idle")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            // Dylib status
            if appState.dylibPath.isEmpty {
                Label("Dylib not found", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundColor(.orange)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color(nsColor: .controlBackgroundColor))
    }
    
    // MARK: - Launcher Panel
    
    private var launcherPanel: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Preset selector
            VStack(alignment: .leading, spacing: 6) {
                Text("Preset")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Picker("", selection: $selectedPreset) {
                    ForEach(0..<presets.count, id: \.self) { i in
                        Text(presets[i].name).tag(i)
                    }
                }
                .pickerStyle(.menu)
                .onChange(of: selectedPreset) { newIndex in
                    let p = presets[newIndex]
                    commandText = p.cmd
                    argsText = p.args
                }
            }
            
            // Command input
            VStack(alignment: .leading, spacing: 6) {
                Text("Command")
                    .font(.caption)
                    .foregroundColor(.secondary)
                HStack {
                    TextField("e.g. python3", text: $commandText)
                        .textFieldStyle(.roundedBorder)
                    Button {
                        let panel = NSOpenPanel()
                        panel.canChooseDirectories = false
                        panel.allowsMultipleSelection = false
                        panel.begin { response in
                            if response == .OK, let url = panel.url {
                                commandText = url.path
                            }
                        }
                    } label: {
                        Image(systemName: "folder")
                    }
                    .buttonStyle(.borderless)
                }
            }
            
            // Arguments input
            VStack(alignment: .leading, spacing: 6) {
                Text("Arguments")
                    .font(.caption)
                    .foregroundColor(.secondary)
                TextField("e.g. script.py --verbose", text: $argsText)
                    .textFieldStyle(.roundedBorder)
            }
            
            // Launch / Stop buttons
            HStack(spacing: 8) {
                Button {
                    let args = argsText.split(separator: " ").map(String.init)
                    appState.launch(command: commandText, arguments: args)
                } label: {
                    Label("Launch", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(appState.isRunning || commandText.isEmpty || appState.dylibPath.isEmpty)
                
                Button {
                    appState.stop()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                .buttonStyle(.bordered)
                .disabled(!appState.isRunning)
            }
            
            Divider()
            
            // How it works
            VStack(alignment: .leading, spacing: 8) {
                Text("How It Works")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                InfoRow(icon: "arrow.triangle.branch", text: "Intercepts malloc/mmap calls")
                InfoRow(icon: "gpu", text: "Compresses with Metal GPU (LZ77)")
                InfoRow(icon: "arrow.down.doc", text: "Decompresses on demand via signal")
                InfoRow(icon: "memorychip", text: "Saves 40-90% physical memory")
            }
            
            Spacer()
            
            // Dylib info
            if !appState.dylibPath.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Dylib")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(appState.dylibPath)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(Color(nsColor: .tertiaryLabelColor))
                        .lineLimit(2)
                }
            }
        }
        .padding(16)
    }
    
    // MARK: - Stats Bar
    
    private var statsBar: some View {
        HStack(spacing: 24) {
            StatBox(title: "Compressed", value: "\(appState.stats.compressions)", unit: "pages")
            StatBox(title: "Saved", value: "\(appState.stats.bytesSaved)", unit: "MB")
            StatBox(title: "Faults", value: "\(appState.stats.faults)", unit: "resolved")
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
    }
    
    // MARK: - Output Panel
    
    private var outputPanel: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    if appState.outputLog.isEmpty {
                        Text("Output will appear here when you launch a process...")
                            .foregroundColor(Color(nsColor: .tertiaryLabelColor))
                            .font(.system(.body, design: .monospaced))
                            .padding(16)
                    }
                    ForEach(appState.outputLog) { entry in
                        HStack(spacing: 6) {
                            Text(entry.time, style: .time)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundColor(Color(nsColor: .quaternaryLabelColor))
                                .frame(width: 60, alignment: .leading)
                            Text(entry.text)
                                .font(.system(.body, design: .monospaced))
                                .foregroundColor(entry.isError ? .red : .primary)
                        }
                        .id(entry.id)
                    }
                }
                .padding(12)
            }
            .onChange(of: appState.outputLog.count) { _ in
                if let last = appState.outputLog.last {
                    proxy.scrollTo(last.id)
                }
            }
        }
    }
}

// MARK: - Subviews

struct InfoRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 11))
                .foregroundColor(.accentColor)
                .frame(width: 16)
            Text(text)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
        }
    }
}

struct StatBox: View {
    let title: String
    let value: String
    let unit: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.system(size: 10))
                .foregroundColor(.secondary)
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(value)
                    .font(.system(size: 18, weight: .semibold, design: .rounded))
                Text(unit)
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(AppState())
        .frame(width: 720, height: 520)
}
