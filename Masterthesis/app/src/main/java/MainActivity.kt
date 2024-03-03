package com.example.master_thesis

import android.Manifest
import android.content.ContentValues.TAG
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.Surface
import android.view.TextureView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.content.res.AssetFileDescriptor
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.ContextCompat
import com.example.master_thesis.R
import com.example.master_thesis.prediction.Classifier
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    private val modelInputSize = 224
    private lateinit var buttonSwitchCamera: Button
    private var currentCameraId: String? = null // To keep track of the currently opened camera
    private lateinit var cameraManager: CameraManager
    private lateinit var textureView: TextureView
    private var cameraDevice: CameraDevice? = null
    private var cameraCaptureSessions: CameraCaptureSession? = null
    private lateinit var captureRequestBuilder: CaptureRequest.Builder
    private lateinit var imageDimension: Size
    private lateinit var classifier: Classifier
    private var monitoringStarted = false
    private var isFrontCameraActive = false // Dodana flaga do śledzenia aktywnej kamery

    private var backgroundHandler: Handler? = null
    private var backgroundThread: HandlerThread? = null

    private lateinit var tfliteInterpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textureView = findViewById(R.id.textureView)
        textureView.surfaceTextureListener = textureListener
        if (allPermissionsGranted()) {
            textureView.surfaceTextureListener = textureListener
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // Initialize TensorFlow Lite interpreter and Classifier
        tfliteInterpreter = Interpreter(loadModelFile("model.tflite"))
        classifier = Classifier(tfliteInterpreter)

        val buttonStartMonitoring: Button = findViewById(R.id.buttonStartMonitoring)
        buttonStartMonitoring.setOnClickListener {
            if (!monitoringStarted) {
                startMonitoring()
                buttonStartMonitoring.text = "Monitoring started"
                monitoringStarted = true
            } else {
                stopMonitoring()
                buttonStartMonitoring.text = "Start monitoring"
                monitoringStarted = false
            }
        }
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        buttonSwitchCamera = findViewById(R.id.buttonSwitchCamera)
        buttonSwitchCamera.setOnClickListener {
            switchCamera()
        }
    }

    private fun switchCamera() {
        // Używamy zmodyfikowanego podejścia do przełączania między kamerami
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            var targetCameraId: String? = null
            for (cameraId in manager.cameraIdList) {
                val characteristics = manager.getCameraCharacteristics(cameraId)
                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (isFrontCameraActive && facing == CameraCharacteristics.LENS_FACING_BACK) {
                    targetCameraId = cameraId
                    break // Znaleziono kamerę tylną, czas na przełączenie
                } else if (!isFrontCameraActive && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    targetCameraId = cameraId
                    break // Znaleziono kamerę przednią, czas na przełączenie
                }
            }

            targetCameraId?.let { cameraId ->
                // Zamykamy obecną sesję kamery
                closeCurrentCamera()

                // Zapisujemy nowy ID kamery i aktualizujemy status kamery
                currentCameraId = cameraId
                isFrontCameraActive = !isFrontCameraActive

                // Otwieramy kamerę z nowym ID
                openCamera(cameraId)
            } ?: run {
                Toast.makeText(this, "Nie znaleziono odpowiedniej kamery.", Toast.LENGTH_SHORT)
                    .show()
            }
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    private fun stopMonitoring() {
        cameraDevice?.close()
        cameraDevice = null
        stopBackgroundThread()
        tfliteInterpreter.close()
        monitoringStarted = false
        Toast.makeText(this, "Monitoring stopped", Toast.LENGTH_SHORT).show()
    }

    //    private fun switchCamera() {
//        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
//        try {
//            for (cameraId in manager.cameraIdList) {
//                val characteristics = manager.getCameraCharacteristics(cameraId)
//                val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
//                if (facing == CameraCharacteristics.LENS_FACING_FRONT) {
//                    // Close the current camera if it's open
//                    cameraDevice?.close()
//                    cameraDevice = null
//                    currentCameraId = cameraId
//                    openCamera()
//                    return // Exit the loop once front-facing camera is found and opened
//                }
//            }
//        } catch (e: CameraAccessException) {
//            e.printStackTrace()
//        }
//    }
    private fun closeCurrentCamera() {
        cameraDevice?.close()
        cameraDevice = null
        cameraCaptureSessions?.close()
        cameraCaptureSessions = null
    }


    private val textureListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            openCamera()
        }

        override fun onSurfaceTextureSizeChanged(
            surface: SurfaceTexture,
            width: Int,
            height: Int
        ) {
        }

        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = false

        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }


    private fun openCamera(cameraId: String? = null) {
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            // Użyj przekazanego ID kamery lub domyślnie otwórz pierwszą kamerę z listy, jeśli nie jest podane.
            val chosenCameraId = cameraId ?: manager.cameraIdList[0]

            val characteristics = manager.getCameraCharacteristics(chosenCameraId)
            val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
            imageDimension = map.getOutputSizes(SurfaceTexture::class.java)[0]
            if (ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.CAMERA),
                    REQUEST_CAMERA_PERMISSION
                )
                return
            }
            manager.openCamera(chosenCameraId, stateCallback, null)
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    private val stateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            cameraDevice = camera
            createCameraPreview()
        }

        override fun onDisconnected(camera: CameraDevice) {
            cameraDevice?.close()
        }

        override fun onError(camera: CameraDevice, error: Int) {
            cameraDevice?.close()
            cameraDevice = null
        }
    }

    //
    protected fun createCameraPreview() {
        try {
            val texture = textureView.surfaceTexture!!
            texture.setDefaultBufferSize(imageDimension.width, imageDimension.height)
            val surface = Surface(texture)
            captureRequestBuilder =
                cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            captureRequestBuilder.addTarget(surface)
            cameraDevice?.createCaptureSession(
                listOf(surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(cameraCaptureSession: CameraCaptureSession) {
                        if (cameraDevice == null) return
                        cameraCaptureSessions = cameraCaptureSession
                        updatePreview()
                    }

                    override fun onConfigureFailed(cameraCaptureSession: CameraCaptureSession) {}
                },
                null
            )
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    //
    private fun startMonitoring() {
        if (cameraDevice == null) {
            openCamera()
        }
        Toast.makeText(applicationContext, "Monitoring startuje", Toast.LENGTH_SHORT).show()

        // Inicjalizacja ImageReader do przechwytywania obrazów z kamery
        val imageReader = ImageReader.newInstance(224, 224, ImageFormat.YUV_420_888, 2)
        imageReader.setOnImageAvailableListener({ reader ->
            val image = reader.acquireLatestImage()
            image?.let {
                val bitmap = convertYuvImageToBitmap(image, 224, 224)
                val result = runModel(bitmap)
                updateUI(result)
                image.close()
            }
        }, backgroundHandler)

        // Dodanie Surface z ImageReader do sesji kamery
        val previewSurface = Surface(textureView.surfaceTexture)
        val recordingSurface = imageReader.surface
        captureRequestBuilder = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            addTarget(previewSurface)
            addTarget(recordingSurface)
        }

        cameraDevice?.createCaptureSession(listOf(previewSurface, recordingSurface), object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                if (cameraDevice == null) return
                cameraCaptureSessions = session
                updatePreview()
            }

            override fun onConfigureFailed(session: CameraCaptureSession) {
                Toast.makeText(this@MainActivity, "Failed to configure camera.", Toast.LENGTH_SHORT).show()
            }
        }, backgroundHandler)
    }

    fun convertYuvImageToBitmap(image: Image, width: Int, height: Int): Bitmap {
        val yBuffer = image.planes[0].buffer // Y
        val uBuffer = image.planes[1].buffer // U
        val vBuffer = image.planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // U and V are swapped
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    fun runModel(bitmap: Bitmap): Float {
        // Załóżmy, że model oczekuje obrazu wejściowego o rozmiarach 224x224 i obraz jest kolorowy (RGB)
        val modelInputSize = 224
        val bitmapScaled = Bitmap.createScaledBitmap(bitmap, modelInputSize, modelInputSize, true)

        // Przygotowanie bufora wejściowego
        val inputImageBuffer = convertBitmapToByteBuffer(bitmapScaled)

        // Przygotuj miejsce na wynik predykcji modelu
        val modelOutput = Array(1) { FloatArray(1) } // Zakładamy, że model zwraca jeden wynik (dla uproszczenia)

        // Uruchom model
        tfliteInterpreter.run(inputImageBuffer, modelOutput)

        // Przetwórz wynik
        val predictionResult = modelOutput[0][0]

        return predictionResult
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * modelInputSize * modelInputSize * 3) // float size * width * height * channels
        byteBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(modelInputSize * modelInputSize)

        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (pixelValue in pixels) {
            val r = (pixelValue shr 16 and 0xFF) / 255.0f
            val g = (pixelValue shr 8 and 0xFF) / 255.0f
            val b = (pixelValue and 0xFF) / 255.0f

            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }

        return byteBuffer
    }

    fun analyzeImage(image: Image) {
        val bitmap = convertYuvImageToBitmap(image, 224, 224) // użyj wcześniej dostarczonej metody
        val prediction = runModel(bitmap)
        runOnUiThread {
            updateUI(prediction)
        }
    }


    private fun convertYuvImageToBitmap(image: Image): Bitmap {
        val buffer = image.planes[0].buffer;
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun updateUI(result: Float) {
        val message = when {
            result > 0.7 -> "The image is classified as: Positive (Confidence: ${String.format("%.2f", result * 100)}%)"
            result > 0.5 -> "The image is classified as: Negative (Confidence: ${String.format("%.2f", result * 100)}%)"
            else -> "Unable to classify the image"
        }
        runOnUiThread {
            findViewById<TextView>(R.id.resultTextView).text = message
        }
    }


    private fun updatePreview() {
        if (cameraDevice == null) return
        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
        try {
            cameraCaptureSessions?.setRepeatingRequest(captureRequestBuilder.build(), null, backgroundHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Error setting up preview", e)
        }
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("Camera Background").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            e.printStackTrace()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                openCamera()
            } else {
                // Handle case where user denies permission
            }
        }
    }

    override fun onResume() {
        super.onResume()
        startBackgroundThread()
        if (textureView.isAvailable) {
            openCamera()
        } else {
            textureView.surfaceTextureListener = textureListener
        }
    }

    override fun onPause() {
        stopBackgroundThread()
        super.onPause()
    }

    companion object {
        private const val REQUEST_CAMERA_PERMISSION = 200
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
}
