import SwiftUI

/// Tiled view for multiple camera streams with overlays
struct MultiCameraView: View {
    let cameras: [CameraConfig]
    let landmarks: [String: DetectionData]
    let frameSize: CGSize

    var body: some View {
        let columns = cameras.count <= 2 ? cameras.count : 2
        let rows = (cameras.count + columns - 1) / columns

        let cellWidth = frameSize.width / CGFloat(columns)
        let cellHeight = frameSize.height / CGFloat(rows)
        let cellSize = CGSize(width: cellWidth, height: cellHeight)

        GeometryReader { geometry in
            VStack(spacing: 0) {
                ForEach(0..<rows, id: \.self) { row in
                    HStack(spacing: 0) {
                        ForEach(0..<columns, id: \.self) { col in
                            let index = row * columns + col
                            if index < cameras.count {
                                let camera = cameras[index]
                                SingleCameraView(
                                    config: camera,
                                    detection: landmarks[camera.id],
                                    frameSize: cellSize
                                )
                                .frame(width: cellSize.width, height: cellSize.height)
                                .clipped()
                            } else {
                                // Empty cell
                                Color.black
                                    .frame(width: cellSize.width, height: cellSize.height)
                            }
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Stats Overlay

struct StatsOverlayView: View {
    let detection: DetectionData?
    let fps: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(String(format: "%.1f FPS", fps))
                .font(.system(size: 12, weight: .medium, design: .monospaced))

            if let detection = detection {
                Text("Frame: \(detection.frameNumber)")
                    .font(.system(size: 10, design: .monospaced))

                Text("Hands: \(detection.hands.count)")
                    .font(.system(size: 10, design: .monospaced))

                if detection.body != nil {
                    Text("Body: detected")
                        .font(.system(size: 10, design: .monospaced))
                }
            }
        }
        .foregroundColor(.white)
        .padding(6)
        .background(Color.black.opacity(0.6))
        .cornerRadius(4)
    }
}

