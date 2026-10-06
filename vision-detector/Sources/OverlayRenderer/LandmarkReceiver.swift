import Foundation

/// Reads JSONL landmark data from stdin asynchronously
final class LandmarkReceiver {
    private let onDetection: (DetectionData) -> Void
    private var isRunning = false
    private let readQueue = DispatchQueue(label: "landmark.reader", qos: .userInteractive)

    init(onDetection: @escaping (DetectionData) -> Void) {
        self.onDetection = onDetection
    }

    func start() {
        guard !isRunning else { return }
        isRunning = true

        readQueue.async { [weak self] in
            self?.readLoop()
        }
    }

    func stop() {
        isRunning = false
    }

    private func readLoop() {
        let handle = FileHandle.standardInput

        while isRunning {
            guard let data = readLine() else {
                // EOF or error
                Thread.sleep(forTimeInterval: 0.01)
                continue
            }

            let line = data.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty else { continue }

            do {
                let detection = try JSONDecoder().decode(DetectionData.self, from: Data(line.utf8))
                onDetection(detection)
            } catch {
                fputs("Warning: Failed to parse landmark JSON: \(error.localizedDescription)\n", stderr)
            }
        }
    }
}

// MARK: - Data Models (matching VisionDetector output)

struct DetectionData: Codable {
    let timestamp: Double
    let frameNumber: Int
    let sourceCamera: String
    let hands: [HandData]
    let body: BodyData?
}

struct HandData: Codable {
    let chirality: String

    // All 21 hand joints
    let wrist: PointData?
    let thumbCMC: PointData?
    let thumbMP: PointData?
    let thumbIP: PointData?
    let thumbTip: PointData?
    let indexMCP: PointData?
    let indexPIP: PointData?
    let indexDIP: PointData?
    let indexTip: PointData?
    let middleMCP: PointData?
    let middlePIP: PointData?
    let middleDIP: PointData?
    let middleTip: PointData?
    let ringMCP: PointData?
    let ringPIP: PointData?
    let ringDIP: PointData?
    let ringTip: PointData?
    let littleMCP: PointData?
    let littlePIP: PointData?
    let littleDIP: PointData?
    let littleTip: PointData?

    /// Get all points as an array for iteration
    var allPoints: [(name: String, point: PointData?)] {
        [
            ("wrist", wrist),
            ("thumbCMC", thumbCMC), ("thumbMP", thumbMP), ("thumbIP", thumbIP), ("thumbTip", thumbTip),
            ("indexMCP", indexMCP), ("indexPIP", indexPIP), ("indexDIP", indexDIP), ("indexTip", indexTip),
            ("middleMCP", middleMCP), ("middlePIP", middlePIP), ("middleDIP", middleDIP), ("middleTip", middleTip),
            ("ringMCP", ringMCP), ("ringPIP", ringPIP), ("ringDIP", ringDIP), ("ringTip", ringTip),
            ("littleMCP", littleMCP), ("littlePIP", littlePIP), ("littleDIP", littleDIP), ("littleTip", littleTip),
        ]
    }

    /// Finger chains for drawing bones (wrist -> tip for each finger)
    static let fingerChains: [[String]] = [
        ["wrist", "thumbCMC", "thumbMP", "thumbIP", "thumbTip"],
        ["wrist", "indexMCP", "indexPIP", "indexDIP", "indexTip"],
        ["wrist", "middleMCP", "middlePIP", "middleDIP", "middleTip"],
        ["wrist", "ringMCP", "ringPIP", "ringDIP", "ringTip"],
        ["wrist", "littleMCP", "littlePIP", "littleDIP", "littleTip"],
    ]

    /// Palm connections (knuckle line)
    static let palmConnections: [[String]] = [
        ["indexMCP", "middleMCP"],
        ["middleMCP", "ringMCP"],
        ["ringMCP", "littleMCP"],
    ]

    func point(named name: String) -> PointData? {
        switch name {
        case "wrist": return wrist
        case "thumbCMC": return thumbCMC
        case "thumbMP": return thumbMP
        case "thumbIP": return thumbIP
        case "thumbTip": return thumbTip
        case "indexMCP": return indexMCP
        case "indexPIP": return indexPIP
        case "indexDIP": return indexDIP
        case "indexTip": return indexTip
        case "middleMCP": return middleMCP
        case "middlePIP": return middlePIP
        case "middleDIP": return middleDIP
        case "middleTip": return middleTip
        case "ringMCP": return ringMCP
        case "ringPIP": return ringPIP
        case "ringDIP": return ringDIP
        case "ringTip": return ringTip
        case "littleMCP": return littleMCP
        case "littlePIP": return littlePIP
        case "littleDIP": return littleDIP
        case "littleTip": return littleTip
        default: return nil
        }
    }
}

struct BodyData: Codable {
    let nose: PointData?
    let leftEye: PointData?
    let rightEye: PointData?
    let leftEar: PointData?
    let rightEar: PointData?
    let neck: PointData?
    let leftShoulder: PointData?
    let rightShoulder: PointData?
    let leftHip: PointData?
    let rightHip: PointData?
    let root: PointData?
    let leftElbow: PointData?
    let leftWrist: PointData?
    let rightElbow: PointData?
    let rightWrist: PointData?
    let leftKnee: PointData?
    let leftAnkle: PointData?
    let rightKnee: PointData?
    let rightAnkle: PointData?

    /// Body skeleton connections
    static let connections: [[String]] = [
        // Head
        ["leftEar", "leftEye"], ["leftEye", "nose"], ["nose", "rightEye"], ["rightEye", "rightEar"],
        // Torso
        ["leftShoulder", "rightShoulder"],
        ["leftShoulder", "leftHip"], ["rightShoulder", "rightHip"],
        ["leftHip", "rightHip"],
        // Left arm
        ["leftShoulder", "leftElbow"], ["leftElbow", "leftWrist"],
        // Right arm
        ["rightShoulder", "rightElbow"], ["rightElbow", "rightWrist"],
        // Left leg
        ["leftHip", "leftKnee"], ["leftKnee", "leftAnkle"],
        // Right leg
        ["rightHip", "rightKnee"], ["rightKnee", "rightAnkle"],
    ]

    func point(named name: String) -> PointData? {
        switch name {
        case "nose": return nose
        case "leftEye": return leftEye
        case "rightEye": return rightEye
        case "leftEar": return leftEar
        case "rightEar": return rightEar
        case "neck": return neck
        case "leftShoulder": return leftShoulder
        case "rightShoulder": return rightShoulder
        case "leftHip": return leftHip
        case "rightHip": return rightHip
        case "root": return root
        case "leftElbow": return leftElbow
        case "leftWrist": return leftWrist
        case "rightElbow": return rightElbow
        case "rightWrist": return rightWrist
        case "leftKnee": return leftKnee
        case "leftAnkle": return leftAnkle
        case "rightKnee": return rightKnee
        case "rightAnkle": return rightAnkle
        default: return nil
        }
    }
}

struct PointData: Codable {
    let x: Double
    let y: Double
    let confidence: Float
}
