import Foundation
import Metal
import MetalKit
import CoreVideo
import simd

/// Vertex data for line segments
struct LineVertex {
    var position: SIMD2<Float>
    var color: SIMD4<Float>
}

/// Uniform data for the render pass
struct RenderUniforms {
    var viewportSize: SIMD2<Float>
}

/// GPU-accelerated overlay renderer using Metal
final class MetalOverlayRenderer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLRenderPipelineState
    private var textureCache: CVMetalTextureCache?

    // Vertex buffers (reused across frames)
    private var lineVertexBuffer: MTLBuffer?
    private var circleVertexBuffer: MTLBuffer?
    private let maxLineVertices = 1024
    private let maxCircleVertices = 2048

    // Circle geometry (pre-computed)
    private let circleSegments = 16

    init?() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fputs("Metal: Failed to create device\n", stderr)
            return nil
        }
        self.device = device

        guard let commandQueue = device.makeCommandQueue() else {
            fputs("Metal: Failed to create command queue\n", stderr)
            return nil
        }
        self.commandQueue = commandQueue

        // Create texture cache for CVPixelBuffer -> MTLTexture conversion
        var cache: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(
            kCFAllocatorDefault,
            nil,
            device,
            nil,
            &cache
        )
        guard status == kCVReturnSuccess, let textureCache = cache else {
            fputs("Metal: Failed to create texture cache\n", stderr)
            return nil
        }
        self.textureCache = textureCache

        // Create render pipeline
        guard let pipelineState = MetalOverlayRenderer.createPipelineState(device: device) else {
            fputs("Metal: Failed to create pipeline state\n", stderr)
            return nil
        }
        self.pipelineState = pipelineState

        // Pre-allocate vertex buffers
        lineVertexBuffer = device.makeBuffer(
            length: maxLineVertices * MemoryLayout<LineVertex>.stride,
            options: .storageModeShared
        )
        circleVertexBuffer = device.makeBuffer(
            length: maxCircleVertices * MemoryLayout<LineVertex>.stride,
            options: .storageModeShared
        )

        fputs("Metal: Renderer initialized successfully\n", stderr)
    }

    private static func createPipelineState(device: MTLDevice) -> MTLRenderPipelineState? {
        // Compile shaders from source
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct VertexIn {
            float2 position [[attribute(0)]];
            float4 color [[attribute(1)]];
        };

        struct VertexOut {
            float4 position [[position]];
            float4 color;
        };

        struct Uniforms {
            float2 viewportSize;
        };

        vertex VertexOut vertexShader(
            VertexIn in [[stage_in]],
            constant Uniforms& uniforms [[buffer(1)]]
        ) {
            VertexOut out;
            // Convert from pixel coordinates to normalized device coordinates
            float2 pixelPos = in.position;
            float2 clipPos = (pixelPos / uniforms.viewportSize) * 2.0 - 1.0;
            clipPos.y = -clipPos.y;  // Flip Y for Metal coordinate system
            out.position = float4(clipPos, 0.0, 1.0);
            out.color = in.color;
            return out;
        }

        fragment float4 fragmentShader(VertexOut in [[stage_in]]) {
            return in.color;
        }
        """

        do {
            let library = try device.makeLibrary(source: shaderSource, options: nil)

            let vertexFunction = library.makeFunction(name: "vertexShader")
            let fragmentFunction = library.makeFunction(name: "fragmentShader")

            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

            // Enable alpha blending
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
            pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
            pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
            pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

            // Vertex descriptor - explicit offsets for LineVertex {position: float2, color: float4}
            let vertexDescriptor = MTLVertexDescriptor()
            vertexDescriptor.attributes[0].format = .float2  // position
            vertexDescriptor.attributes[0].offset = 0
            vertexDescriptor.attributes[0].bufferIndex = 0
            vertexDescriptor.attributes[1].format = .float4  // color
            vertexDescriptor.attributes[1].offset = 8  // sizeof(float2) = 8 bytes
            vertexDescriptor.attributes[1].bufferIndex = 0
            vertexDescriptor.layouts[0].stride = 24  // sizeof(float2) + sizeof(float4) = 8 + 16 = 24
            pipelineDescriptor.vertexDescriptor = vertexDescriptor

            return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            fputs("Metal: Failed to create pipeline: \(error)\n", stderr)
            return nil
        }
    }

    /// Render skeleton overlay onto the pixel buffer
    func renderOverlay(
        onto pixelBuffer: CVPixelBuffer,
        hands: [HandLandmarks],
        body: BodyLandmarks?,
        minConfidence: Float
    ) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        // Create Metal texture from pixel buffer
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache!,
            pixelBuffer,
            nil,
            .bgra8Unorm,
            width,
            height,
            0,
            &cvTexture
        )

        guard status == kCVReturnSuccess,
              let cvTex = cvTexture,
              let texture = CVMetalTextureGetTexture(cvTex) else {
            fputs("Metal: Failed to create texture from pixel buffer\n", stderr)
            return
        }

        // Build vertex data for lines and circles
        var lineVertices: [LineVertex] = []
        var circleVertices: [LineVertex] = []

        let w = Float(width)
        let h = Float(height)

        // Draw hands
        for (handIndex, hand) in hands.enumerated() {
            buildHandGeometry(
                hand: hand,
                width: w,
                height: h,
                minConfidence: minConfidence,
                colorIndex: handIndex,
                lineVertices: &lineVertices,
                circleVertices: &circleVertices
            )
        }

        // Draw body
        if let body = body {
            buildBodyGeometry(
                body: body,
                width: w,
                height: h,
                minConfidence: minConfidence,
                lineVertices: &lineVertices,
                circleVertices: &circleVertices
            )
        }

        // Upload vertex data
        if !lineVertices.isEmpty {
            lineVertices.withUnsafeBytes { ptr in
                lineVertexBuffer?.contents().copyMemory(from: ptr.baseAddress!, byteCount: ptr.count)
            }
        }
        if !circleVertices.isEmpty {
            circleVertices.withUnsafeBytes { ptr in
                circleVertexBuffer?.contents().copyMemory(from: ptr.baseAddress!, byteCount: ptr.count)
            }
        }

        // Create command buffer and render
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = texture
        renderPassDescriptor.colorAttachments[0].loadAction = .load  // Keep existing content
        renderPassDescriptor.colorAttachments[0].storeAction = .store

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }

        encoder.setRenderPipelineState(pipelineState)

        // Set uniforms
        var uniforms = RenderUniforms(viewportSize: SIMD2<Float>(w, h))
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<RenderUniforms>.stride, index: 1)

        // Draw lines
        if !lineVertices.isEmpty {
            encoder.setVertexBuffer(lineVertexBuffer, offset: 0, index: 0)
            encoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: lineVertices.count)
        }

        // Draw circles (as triangle fans)
        if !circleVertices.isEmpty {
            encoder.setVertexBuffer(circleVertexBuffer, offset: 0, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: circleVertices.count)
        }

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Flush texture cache
        CVMetalTextureCacheFlush(textureCache!, 0)
    }

    private func buildHandGeometry(
        hand: HandLandmarks,
        width: Float,
        height: Float,
        minConfidence: Float,
        colorIndex: Int,
        lineVertices: inout [LineVertex],
        circleVertices: inout [LineVertex]
    ) {
        let colors: [SIMD4<Float>] = [
            SIMD4<Float>(0, 1, 0, 1),       // Green - thumb
            SIMD4<Float>(0, 1, 1, 1),       // Cyan - index
            SIMD4<Float>(0, 0.5, 1, 1),     // Blue - middle
            SIMD4<Float>(1, 0.5, 0.8, 1),   // Pink - ring
            SIMD4<Float>(1, 0.3, 0.3, 1),   // Red - little
        ]

        let fingerChains: [[KeyPath<HandLandmarks, Point2D?>]] = [
            [\.wrist, \.thumbCMC, \.thumbMP, \.thumbIP, \.thumbTip],
            [\.wrist, \.indexMCP, \.indexPIP, \.indexDIP, \.indexTip],
            [\.wrist, \.middleMCP, \.middlePIP, \.middleDIP, \.middleTip],
            [\.wrist, \.ringMCP, \.ringPIP, \.ringDIP, \.ringTip],
            [\.wrist, \.littleMCP, \.littlePIP, \.littleDIP, \.littleTip],
        ]

        // Build line segments for each finger
        for (fingerIdx, chain) in fingerChains.enumerated() {
            let color = colors[fingerIdx % colors.count]
            var prevPoint: SIMD2<Float>? = nil

            for keyPath in chain {
                guard let point = hand[keyPath: keyPath], point.confidence >= minConfidence else { continue }

                let x = Float(point.x) * width
                let y = Float(1.0 - point.y) * height  // Flip Y
                let pos = SIMD2<Float>(x, y)

                if let prev = prevPoint {
                    lineVertices.append(LineVertex(position: prev, color: color))
                    lineVertices.append(LineVertex(position: pos, color: color))
                }
                prevPoint = pos
            }
        }

        // Build circles for joints
        let jointColor = SIMD4<Float>(1, 1, 1, 1)  // White
        let allJoints: [KeyPath<HandLandmarks, Point2D?>] = [
            \.wrist,
            \.thumbCMC, \.thumbMP, \.thumbIP, \.thumbTip,
            \.indexMCP, \.indexPIP, \.indexDIP, \.indexTip,
            \.middleMCP, \.middlePIP, \.middleDIP, \.middleTip,
            \.ringMCP, \.ringPIP, \.ringDIP, \.ringTip,
            \.littleMCP, \.littlePIP, \.littleDIP, \.littleTip,
        ]

        for keyPath in allJoints {
            guard let point = hand[keyPath: keyPath], point.confidence >= minConfidence else { continue }

            let x = Float(point.x) * width
            let y = Float(1.0 - point.y) * height
            addCircle(at: SIMD2<Float>(x, y), radius: 4, color: jointColor, to: &circleVertices)
        }
    }

    private func buildBodyGeometry(
        body: BodyLandmarks,
        width: Float,
        height: Float,
        minConfidence: Float,
        lineVertices: inout [LineVertex],
        circleVertices: inout [LineVertex]
    ) {
        let color = SIMD4<Float>(1, 1, 0, 1)  // Yellow

        let connections: [(KeyPath<BodyLandmarks, Point2D?>, KeyPath<BodyLandmarks, Point2D?>)] = [
            // Head
            (\.leftEar, \.leftEye), (\.leftEye, \.nose), (\.nose, \.rightEye), (\.rightEye, \.rightEar),
            // Torso
            (\.leftShoulder, \.rightShoulder),
            (\.leftShoulder, \.leftHip), (\.rightShoulder, \.rightHip),
            (\.leftHip, \.rightHip),
            // Arms
            (\.leftShoulder, \.leftElbow), (\.leftElbow, \.leftWrist),
            (\.rightShoulder, \.rightElbow), (\.rightElbow, \.rightWrist),
            // Legs
            (\.leftHip, \.leftKnee), (\.leftKnee, \.leftAnkle),
            (\.rightHip, \.rightKnee), (\.rightKnee, \.rightAnkle),
        ]

        for (from, to) in connections {
            guard let p1 = body[keyPath: from], p1.confidence >= minConfidence,
                  let p2 = body[keyPath: to], p2.confidence >= minConfidence else { continue }

            let pos1 = SIMD2<Float>(Float(p1.x) * width, Float(1.0 - p1.y) * height)
            let pos2 = SIMD2<Float>(Float(p2.x) * width, Float(1.0 - p2.y) * height)

            lineVertices.append(LineVertex(position: pos1, color: color))
            lineVertices.append(LineVertex(position: pos2, color: color))
        }

        // Draw joints
        let joints: [KeyPath<BodyLandmarks, Point2D?>] = [
            \.nose, \.leftEye, \.rightEye, \.leftEar, \.rightEar,
            \.leftShoulder, \.rightShoulder, \.leftElbow, \.rightElbow,
            \.leftWrist, \.rightWrist, \.leftHip, \.rightHip,
            \.leftKnee, \.rightKnee, \.leftAnkle, \.rightAnkle,
        ]

        for keyPath in joints {
            guard let point = body[keyPath: keyPath], point.confidence >= minConfidence else { continue }

            let x = Float(point.x) * width
            let y = Float(1.0 - point.y) * height
            addCircle(at: SIMD2<Float>(x, y), radius: 5, color: color, to: &circleVertices)
        }
    }

    private func addCircle(at center: SIMD2<Float>, radius: Float, color: SIMD4<Float>, to vertices: inout [LineVertex]) {
        // Build circle as triangle fan
        for i in 0..<circleSegments {
            let angle1 = Float(i) / Float(circleSegments) * 2 * .pi
            let angle2 = Float(i + 1) / Float(circleSegments) * 2 * .pi

            let p1 = SIMD2<Float>(
                center.x + cos(angle1) * radius,
                center.y + sin(angle1) * radius
            )
            let p2 = SIMD2<Float>(
                center.x + cos(angle2) * radius,
                center.y + sin(angle2) * radius
            )

            vertices.append(LineVertex(position: center, color: color))
            vertices.append(LineVertex(position: p1, color: color))
            vertices.append(LineVertex(position: p2, color: color))
        }
    }
}
