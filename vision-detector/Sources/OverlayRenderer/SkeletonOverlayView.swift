import SwiftUI

/// Renders hand and body skeleton overlay using SwiftUI Canvas
struct SkeletonOverlayView: View {
    let detection: DetectionData
    let frameSize: CGSize
    let minConfidence: Float

    // Finger colors (FreiHAND scheme)
    static let fingerColors: [Color] = [
        .green,   // Thumb
        .cyan,    // Index
        .blue,    // Middle
        .pink,    // Ring
        .red,     // Little
    ]

    init(detection: DetectionData, frameSize: CGSize, minConfidence: Float = 0.3) {
        self.detection = detection
        self.frameSize = frameSize
        self.minConfidence = minConfidence
    }

    var body: some View {
        Canvas { context, size in
            // Draw body skeleton
            if let body = detection.body {
                drawBody(body, context: &context, size: size)
            }

            // Draw hand skeletons
            for hand in detection.hands {
                drawHand(hand, context: &context, size: size)
            }
        }
        .frame(width: frameSize.width, height: frameSize.height)
    }

    // MARK: - Hand Drawing

    private func drawHand(_ hand: HandData, context: inout GraphicsContext, size: CGSize) {
        // Draw finger bones
        for (fingerIndex, chain) in HandData.fingerChains.enumerated() {
            let color = Self.fingerColors[fingerIndex]
            drawChain(chain, hand: hand, color: color, context: &context, size: size)
        }

        // Draw palm connections (knuckle line)
        for connection in HandData.palmConnections {
            drawConnection(connection[0], connection[1], hand: hand, color: .white.opacity(0.7), context: &context, size: size)
        }

        // Draw joints
        for (_, point) in hand.allPoints {
            if let p = point, p.confidence >= minConfidence {
                let screenPoint = toScreenCoords(p, size: size)
                let radius: CGFloat = 4

                // Outer glow
                let glowRect = CGRect(
                    x: screenPoint.x - radius - 2,
                    y: screenPoint.y - radius - 2,
                    width: (radius + 2) * 2,
                    height: (radius + 2) * 2
                )
                context.fill(
                    Circle().path(in: glowRect),
                    with: .color(.white.opacity(0.3))
                )

                // Inner joint
                let jointRect = CGRect(
                    x: screenPoint.x - radius,
                    y: screenPoint.y - radius,
                    width: radius * 2,
                    height: radius * 2
                )
                context.fill(
                    Circle().path(in: jointRect),
                    with: .color(.white)
                )
            }
        }
    }

    private func drawChain(_ chain: [String], hand: HandData, color: Color, context: inout GraphicsContext, size: CGSize) {
        var path = Path()
        var isFirst = true

        for jointName in chain {
            guard let point = hand.point(named: jointName),
                  point.confidence >= minConfidence else { continue }

            let screenPoint = toScreenCoords(point, size: size)

            if isFirst {
                path.move(to: screenPoint)
                isFirst = false
            } else {
                path.addLine(to: screenPoint)
            }
        }

        context.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
    }

    private func drawConnection(_ from: String, _ to: String, hand: HandData, color: Color, context: inout GraphicsContext, size: CGSize) {
        guard let p1 = hand.point(named: from), p1.confidence >= minConfidence,
              let p2 = hand.point(named: to), p2.confidence >= minConfidence else { return }

        var path = Path()
        path.move(to: toScreenCoords(p1, size: size))
        path.addLine(to: toScreenCoords(p2, size: size))

        context.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: 2, lineCap: .round))
    }

    // MARK: - Body Drawing

    private func drawBody(_ body: BodyData, context: inout GraphicsContext, size: CGSize) {
        let bodyColor = Color.yellow

        // Draw skeleton connections
        for connection in BodyData.connections {
            guard let p1 = body.point(named: connection[0]), p1.confidence >= minConfidence,
                  let p2 = body.point(named: connection[1]), p2.confidence >= minConfidence else { continue }

            var path = Path()
            path.move(to: toScreenCoords(p1, size: size))
            path.addLine(to: toScreenCoords(p2, size: size))

            context.stroke(path, with: .color(bodyColor), style: StrokeStyle(lineWidth: 2, lineCap: .round))
        }

        // Draw joints
        let bodyJoints: [PointData?] = [
            body.nose, body.leftEye, body.rightEye, body.leftEar, body.rightEar,
            body.leftShoulder, body.rightShoulder, body.leftElbow, body.rightElbow,
            body.leftWrist, body.rightWrist, body.leftHip, body.rightHip,
            body.leftKnee, body.rightKnee, body.leftAnkle, body.rightAnkle
        ]

        for point in bodyJoints {
            if let p = point, p.confidence >= minConfidence {
                let screenPoint = toScreenCoords(p, size: size)
                let radius: CGFloat = 5

                let jointRect = CGRect(
                    x: screenPoint.x - radius,
                    y: screenPoint.y - radius,
                    width: radius * 2,
                    height: radius * 2
                )
                context.fill(Circle().path(in: jointRect), with: .color(bodyColor))
            }
        }
    }

    // MARK: - Coordinate Conversion

    /// Convert Vision normalized coordinates (origin lower-left) to screen coordinates (origin upper-left)
    private func toScreenCoords(_ point: PointData, size: CGSize) -> CGPoint {
        CGPoint(
            x: point.x * size.width,
            y: (1.0 - point.y) * size.height  // Flip Y axis
        )
    }
}

