import Foundation
import UIKit
import Vision
import TensorSwift
import AVFoundation

let posenet = PoseNet()
var isXcode : Bool = false // true: localfile , false: device camera

// controlling the pace of the machine vision analysis
var lastAnalysis: TimeInterval = 0
var pace: TimeInterval = 0.08 // in seconds, classification will not repeat faster than this value
// performance tracking
let trackPerformance = true // use "true" for performance logging
var frameCount = 0
let framesPerSample = 10
var startDate = NSDate.timeIntervalSinceReferenceDate
let semaphore = DispatchSemaphore(value: 1)

class ViewController: UIViewController {
    
    @IBOutlet weak var previewView: UIImageView!
    @IBOutlet weak var lineView: UIImageView!
    
    // 初始化model
    let model = stylize()
    //  根据model设定imageSize
    let targetImageSize = CGSize(width: 256, height: 256)
    // 摄像头
    var previewLayer: AVCaptureVideoPreviewLayer!
    var request: VNCoreMLRequest!
    // 视频队列
    let videoQueue = DispatchQueue(label: "videoQueue")
    // 绘制队列
    let drawQueue = DispatchQueue(label: "drawQueue")
    
    // 拍摄session
    var captureSession = AVCaptureSession()
    var captureDevice: AVCaptureDevice?
    let videoOutput = AVCaptureVideoDataOutput()
    var isWriting : Bool = false
    
    var changeIndex = 0
    let imageView = UIImageView()
    let changeButton = UIButton()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // 设置预览View
        previewView.frame = UIScreen.main.bounds
        previewView.contentMode = .scaleAspectFit
        imageView.frame = CGRect(x: 0, y: 0, width: UIScreen.main.bounds.size.width, height: UIScreen.main.bounds.size.width)
        changeButton.frame = CGRect(x: 0, y: UIScreen.main.bounds.size.width + 100, width: UIScreen.main.bounds.size.width, height: 100)
        changeButton.addTarget(self, action: #selector(self.pressed(sender:)), for: .touchUpInside)
        if (isXcode){
            // 处理图片
            let fname = "soccer"
            if let image = UIImage(named: fname)?.resize(to: targetImageSize){
                previewView.image = image
                // coreML入口
                let result = measure(// 计时
                    runCoreML(// 姿态识别函数 输入数据是resize之后的pixelBuffer
                        image.pixelBufferGray(width: 96, height: 96)!
//                        (image.grayImage()?.pixelBuffer()!)!
                    )
                )
                
                print(result.duration)// 输出耗时
                drawResults(result.result)// 将结果绘制到UIImage上
                //let result = runOffline()
                //drawResults(result)
            }
        } else {
            // 摄像头数据
            previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
            previewView.layer.addSublayer(previewLayer)
        }
        
        
//        lazy imageView = UIImageView(frame: CGRect(x: 50, y: 50, width: 200, height: 200));
        self.view.addSubview(imageView)
        self.view.addSubview(changeButton)
    }
    
    @objc func pressed(sender: UIButton!) {
        self.changeIndex+=1
    }
    
    override func viewDidAppear(_ animated: Bool) {
        if (!isXcode){
            setupCamera()
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        if (!isXcode){
            previewLayer.frame = previewView.bounds;
            lineView.frame = previewView.bounds;
        }
    }
    

    
    func setupCamera() {
        let deviceDiscovery = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .back)

        if let device = deviceDiscovery.devices.last {
            captureDevice = device
            beginSession()
        }
    }
    
    func beginSession() {
        do {
            videoOutput.videoSettings = [((kCVPixelBufferPixelFormatTypeKey as NSString) as String) : (NSNumber(value: kCVPixelFormatType_32BGRA) as! UInt32)]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            
            if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiom.phone) {
//                captureSession.sessionPreset = .hd1920x1080
                captureSession.sessionPreset = .photo
            } else if (UI_USER_INTERFACE_IDIOM() == UIUserInterfaceIdiom.pad) {
                captureSession.sessionPreset = .photo
            }
            
            captureSession.addOutput(videoOutput)
            
            let input = try AVCaptureDeviceInput(device: captureDevice!)
            captureSession.addInput(input)
            
            captureSession.startRunning()
        } catch {
            print("error connecting to capture device")
        }
    }
    
    func drawResults(_ poses: [Pose]){
        
        let minPoseConfidence: Float = 0.5 // ?? 什么意思？识别出的姿态的分数？可信度大于0.5的结果才被绘制上去 否则不绘制
        
        let screen = UIScreen.main.bounds
        let scale = screen.width / self.targetImageSize.width
        let size = AVMakeRect(aspectRatio: self.targetImageSize,
                              insideRect: self.previewView.frame)
        
        var linePath = UIBezierPath()
        var arcPath = UIBezierPath()
        // 将每个pose结果绘制出来如果pose的可信度高
        poses.forEach { pose in
            if (pose.score >= minPoseConfidence){
                self.drawKeypoints(arcPath: &arcPath, keypoints: pose.keypoints,minConfidence: minPoseConfidence,
                                   size: size.origin, scale: scale)// 绘制关键点
                self.drawSkeleton(linePath: &linePath, keypoints: pose.keypoints,
                                  minConfidence: minPoseConfidence,
                                  size: size.origin, scale: scale)// 绘制连接线
            }
        }
        
        // Draw
        let arcLine = CAShapeLayer()
        arcLine.path = arcPath.cgPath
        arcLine.strokeColor = UIColor.green.cgColor
        
        let line = CAShapeLayer()
        line.path = linePath.cgPath
        line.strokeColor = UIColor.red.cgColor
        line.lineWidth = 2
        line.lineJoin = kCALineJoinRound
        
        self.lineView.layer.sublayers = nil
        self.lineView.layer.addSublayer(arcLine)
        self.lineView.layer.addSublayer(line)
        linePath.removeAllPoints()
        arcPath.removeAllPoints()
        semaphore.wait()
        isWriting = false
        semaphore.signal()
        
    }
    
    func drawKeypoints(arcPath: inout UIBezierPath, keypoints: [Keypoint], minConfidence: Float,
                       size: CGPoint,scale: CGFloat = 1){
        
        keypoints.forEach { keypoint in
            if (keypoint.score < minConfidence) {
                return
            }
            let center = CGPoint(x: CGFloat(keypoint.position.x) * scale + size.x,
                                 y: CGFloat(keypoint.position.y) * scale + size.y)
            let trackPath = UIBezierPath(arcCenter: center,
                                         radius: 3, startAngle: 0,
                                         endAngle: 2.0 * .pi, clockwise: true)
            
            arcPath.append(trackPath)
        }
    }
    
    func drawSegment(linePath: inout UIBezierPath,fromPoint start: CGPoint, toPoint end:CGPoint,
                     size: CGPoint, scale: CGFloat = 1) {
        
        let newlinePath = UIBezierPath()
        newlinePath.move(to:
            CGPoint(x: start.x * scale + size.x, y: start.y * scale + size.y))
        newlinePath.addLine(to:
            CGPoint(x: end.x * scale + size.x, y: end.y * scale + size.y))
        linePath.append(newlinePath)
    }
    func drawSkeleton(linePath: inout UIBezierPath,keypoints: [Keypoint], minConfidence: Float,
                      size: CGPoint, scale: CGFloat = 1){
        let adjacentKeyPoints = getAdjacentKeyPoints(
            keypoints: keypoints, minConfidence: minConfidence);
        
        adjacentKeyPoints.forEach { keypoint in
            drawSegment(linePath: &linePath,
                        fromPoint:
                CGPoint(x: CGFloat(keypoint[0].position.x),y: CGFloat(keypoint[0].position.y)),
                        toPoint:
                CGPoint(x: CGFloat(keypoint[1].position.x),y: CGFloat(keypoint[1].position.y)),
                        size: size,
                        scale: scale
            )
        }
    }
    
    func eitherPointDoesntMeetConfidence(
        _ a: Float,_ b: Float,_ minConfidence: Float) -> Bool {
        return (a < minConfidence || b < minConfidence)
    }
    
    func getAdjacentKeyPoints(
        keypoints: [Keypoint], minConfidence: Float)-> [[Keypoint]] {
        
        return connectedPartIndices.filter {
            !eitherPointDoesntMeetConfidence(
                keypoints[$0.0].score,
                keypoints[$0.1].score,
                minConfidence)
            }.map { [keypoints[$0.0],keypoints[$0.1]] }
    }
    
    
    // 离线跑是个什么意思？项目中的bin文件应该确定是在别的地方跑出来的数据存储进项目测试用的！
    func runOffline() -> [Pose]{
        
        let scores = getTensorTranspose("heatmapScores",[33, 33, 17])
        let offsets = getTensorTranspose("offsets",[33, 33, 34])
        let displacementsFwd = getTensorTranspose("displacementsFwd",[33, 33, 32])
        let displacementsBwd = getTensorTranspose("displacementsBwd",[33, 33, 32])
        
        let sum = scores.reduce(0, +) / (17 * 33 * 33)
        print(sum)
        
        let poses = posenet.decodeMultiplePoses(
            scores: scores,
            offsets: offsets,
            displacementsFwd: displacementsFwd,
            displacementsBwd: displacementsBwd,
            outputStride: 16, maxPoseDetections: 5,
            scoreThreshold: 0.5,nmsRadius: 20)
        
        return poses
    }
    
    func runCoreML(_ img: CVPixelBuffer) -> [Pose]{
        let numStyles  = 26
        
        let styleArray = try? MLMultiArray(shape: [numStyles,1,1,1,1] as [NSNumber], dataType: MLMultiArrayDataType.double)
        
        for i in 0...(numStyles - 1) {
            styleArray?[i] = 0.0
        }
        styleArray?[self.changeIndex % 26] = 1.0
        
        let options = MLPredictionOptions()
        options.usesCPUOnly = true
        
        let input = stylizeInput(style_num__0: styleArray!, input__0: img)
        var image_1 = UIImage.init()
        do {
            let prediction = try model.prediction(input: input, options: options)
            image_1 = UIImage(pixelBuffer: prediction.Squeeze__0)!
        } catch {
            //handle error
            print(error)
        }
        
//            try self.Context!.executeFetchRequest(request) as! [AccountDetail]
        
//        if let pixelBuffer = img {
//            NSLog("1")
//            let options = MLPredictionOptions()
//            options.usesCPUOnly = true
//            let input = stylizeInput(style_num__0: styleArray!, input__0: pixelBuffer)
//            //                input.style_num__0 = styleArray!
//            //                let prediction = input.input__0 = pixelBuffer!
//            let prediction =  try model.prediction(input: input, options: options)
//            //                let prediction = try model.prediction(style_num__0: styleArray!, input__0: pixelBuffer,options:options)
//            NSLog("2")
//            let image_1 = UIImage(pixelBuffer: prediction.Squeeze__0)!
//            UIImageWriteToSavedPhotosAlbum(image_1, nil, nil, nil)
//            self.userImage = image_1
//        }
//        let result = try? model.prediction(image: img)
//        let image = UIImage(cgImage: MultiArray<Double>(pp.featureValue.multiArrayValue!).reshaped([2, 224, 224]).image(channel: 1, offset: 0, scale: 255)!.cgImage!, scale: 1.0, orientation: UIImageOrientation.upMirrored)
//        if (result != nil) {
//            print(result?.featureValue(for: "output1")?.multiArrayValue)
//            result?.featureValue(for: "output1")?.multiArrayValue
//            result?.featureValue(for: "output1")?.multiArrayValue!)!
//            let image = UIImage(cgImage: MultiArray<Double>((result?.featureValue(for: "output1")?.multiArrayValue!)!).reshaped([1,96,96]).image(channel: 0, offset: 0, scale: 255)!.cgImage!)
            DispatchQueue.main.async {
                self.imageView.image = image_1
                UIImageWriteToSavedPhotosAlbum(image_1, nil, nil, nil)
            }
//        }
//        let image = UIImage(cgImage: MultiArray<Double>((result?.featureValue(for: "output1")?.multiArrayValue!)!).reshaped([1,96,96]).image(channel: 0, offset: 0, scale: 255)!.cgImage!)
////        result?.featureValue(for: "")
//        DispatchQueue.main.async {
//            self.imageView.image = image
//        }


        return []
    }
    
    func processPixels(cvImage: UIImage, userImage:UIImage) -> UIImage? {
        
        //guard GameManager.shared.isGameStart else { return nil }
        
        let inputCGImage = cvImage.cgImage!
        
        
        /*// Create the CIImages from a file
         var userCIImage = CIImage(cgImage: newCGImage)
         var alphaCIImage = CIImage(cgImage: inputCGImage)
         
         // Create a CIBlendWithAlphaMask filter with our three input images
         var blend_with_alpha_mask = new CIBlendWithAlphaMask () {
         BackgroundImage = clouds,
         Image = flower,
         Mask = xamarinAlpha
         };
         
         // Get the blended image from the filter
         var output = blend_with_alpha_mask.OutputImage;
         
         // To render the results, we need to create a context, and then
         // use one of the context rendering APIs, in this case, we render the
         // result into a CoreGraphics image, which is merely a useful representation
         //
         var context = CIContext.FromOptions (null);
         
         var cgimage = context.CreateCGImage (output, output.Extent);
         
         // The above cgimage can be added to a screen view, for example, this
         // would add it to a UIImageView on the screen:
         myImageView.Image = UIImage.FromImage (cgimage);*/
        
        
        let colorSpace       = CGColorSpaceCreateDeviceRGB()
        let width            = inputCGImage.width
        let height           = inputCGImage.height
        let bytesPerPixel    = 4
        let bitsPerComponent = 8
        let bytesPerRow      = bytesPerPixel * width
        let bitmapInfo       = RGBA32.bitmapInfo
        
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo) else {
            print("unable to create context")
            return nil
        }
        
        guard let context_2 = CGContext(data: nil, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo) else {
            print("unable to create context")
            return nil
        }
        
        context.draw(inputCGImage, in: CGRect(x: 0, y: 0, width: width, height: height))
//        context_2.draw(newCGImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let buffer = context.data, let buffer_2 = context_2.data else {
            print("unable to get context data")
            return nil
        }
        
        let pixelBuffer = buffer.bindMemory(to: RGBA32.self, capacity: width * height)
//        let pixelBuffer_2 = buffer_2.bindMemory(to: RGBA32.self, capacity: width * height)
        
        for row in 0 ..< Int(height) {
            for column in 0 ..< Int(width) {
                let offset = row * width + column
                
                let alpha = UInt32(pixelBuffer[offset].redComponent)
                if alpha <= 128
                {
                    //pixelBuffer_2[offset] = .clear
                    //pixelBuffer_2[offset] = .alpha128
                    
                    if alpha > 75
                    {
                        pixelBuffer[offset] = RGBA32(rgb:pixelBuffer[offset], alpha:alpha)
                    }
                    else
                    {
                        pixelBuffer[offset] = .white
                    }
                }
                
                
                
                
            }
        }
        
        
        
        let outputCGImage = context.makeImage()!
        let outputImage = UIImage(cgImage: outputCGImage, scale: cvImage.scale, orientation: cvImage.imageOrientation)
        
        return outputImage
    }

}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    // called for each frame of video
    func captureOutput(_ captureOutput: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        let currentDate = NSDate.timeIntervalSinceReferenceDate
        
        // control the pace of the machine vision to protect battery life
        if currentDate - lastAnalysis >= pace {
            lastAnalysis = currentDate
        } else {
            return // don't run the classifier more often than we need
        }
        
        // keep track of performance and log the frame rate
        if trackPerformance {
            frameCount = frameCount + 1
            if frameCount % framesPerSample == 0 {
                let diff = currentDate - startDate
                if (diff > 0) {
                    if pace > 0.0 {
                        print("WARNING: Frame rate of image classification is being limited by \"pace\" setting. Set to 0.0 for fastest possible rate.")
                    }
                    print("\(String.localizedStringWithFormat("%0.2f", (diff/Double(framesPerSample))))s per frame (average)")
                }
                startDate = currentDate
            }
        }
        
//        DispatchQueue.global(qos: .default).async {
        drawQueue.async {
            semaphore.wait()
            if (self.isWriting == false) {
                self.isWriting = true
                semaphore.signal()
                let startTime = CFAbsoluteTimeGetCurrent()
                guard let croppedBuffer = croppedSampleBuffer(sampleBuffer, targetSize: self.targetImageSize) else {
                    return
                }
                let poses = self.runCoreML(croppedBuffer)
                DispatchQueue.main.sync {
                    self.drawResults(poses)
                }
                let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
                print ("Elapsed time is \(timeElapsed) seconds.")
            } else {
                semaphore.signal()
            }
        }
    }
}

let context = CIContext()
var rotateTransform: CGAffineTransform?
var scaleTransform: CGAffineTransform?
var cropTransform: CGAffineTransform?
var resultBuffer: CVPixelBuffer?

func croppedSampleBuffer(_ sampleBuffer: CMSampleBuffer, targetSize: CGSize) -> CVPixelBuffer? {
    
    guard let imageBuffer: CVImageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
        fatalError("Can't convert to CVImageBuffer.")
    }
    
    
    // Only doing these calculations once for efficiency.
    // If the incoming images could change orientation or size during a session, this would need to be reset when that happens.
    if rotateTransform == nil {
        let imageSize = CVImageBufferGetEncodedSize(imageBuffer)
        let rotatedSize = CGSize(width: imageSize.height, height: imageSize.width)
        
        guard targetSize.width < rotatedSize.width, targetSize.height < rotatedSize.height else {
            fatalError("Captured image is smaller than image size for model.")
        }
        
        let shorterSize = (rotatedSize.width < rotatedSize.height) ? rotatedSize.width : rotatedSize.height
        rotateTransform = CGAffineTransform(translationX: imageSize.width / 2.0, y: imageSize.height / 2.0).rotated(by: -CGFloat.pi / 2.0).translatedBy(x: -imageSize.height / 2.0, y: -imageSize.width / 2.0)
        
        let scale = targetSize.width / shorterSize
        scaleTransform = CGAffineTransform(scaleX: scale, y: scale)
        
        // Crop input image to output size
        let xDiff = rotatedSize.width * scale - targetSize.width
        let yDiff = rotatedSize.height * scale - targetSize.height
        cropTransform = CGAffineTransform(translationX: xDiff/2.0, y: yDiff/2.0)
    }
    
    // Convert to CIImage because it is easier to manipulate
    let ciImage = CIImage(cvImageBuffer: imageBuffer)
    let rotated = ciImage.transformed(by: rotateTransform!)
    let scaled = rotated.transformed(by: scaleTransform!)
    let cropped = scaled.transformed(by: cropTransform!)
    
    // Note that the above pipeline could be easily appended with other image manipulations.
    // For example, to change the image contrast. It would be most efficient to handle all of
    // the image manipulation in a single Core Image pipeline because it can be hardware optimized.
    
    // Only need to create this buffer one time and then we can reuse it for every frame
    if resultBuffer == nil {
        let result = CVPixelBufferCreate(kCFAllocatorDefault, Int(targetSize.width), Int(targetSize.height), kCVPixelFormatType_32BGRA, nil, &resultBuffer)
        
        guard result == kCVReturnSuccess else {
            fatalError("Can't allocate pixel buffer.")
        }
    }
    
    // Render the Core Image pipeline to the buffer
    context.render(cropped, to: resultBuffer!)
    
    //  For debugging
    //  let image = imageBufferToUIImage(resultBuffer!)
    //  print(image.size) // set breakpoint to see image being provided to CoreML
    
    return resultBuffer
}

