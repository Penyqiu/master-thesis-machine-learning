package com.example.master_thesis.prediction

import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Classifier(private val interpreter: Interpreter) {

    private val imageSizeX = 224 // szerokość obrazu oczekiwana przez model
    private val imageSizeY = 224 // wysokość obrazu oczekiwana przez model

    // Metoda do zmiany rozmiaru bitmapy do rozmiaru oczekiwanego przez model
    private fun Bitmap.resize(imageSizeX: Int, imageSizeY: Int): Bitmap {
        return Bitmap.createScaledBitmap(this, imageSizeX, imageSizeY, true)
    }

    // Metoda do konwersji bitmapy do ByteBuffer
    private fun Bitmap.toByteBuffer(): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * imageSizeX * imageSizeY * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(imageSizeX * imageSizeY)
        this.getPixels(pixels, 0, width, 0, 0, width, height)

        for (pixelValue in pixels) {
            val r = (pixelValue shr 16 and 0xFF)
            val g = (pixelValue shr 8 and 0xFF)
            val b = (pixelValue and 0xFF)

            // Konwersja RGB do skali szarości (prosty przykład)
            val normalizedPixelValue = (r + g + b) / 3.0f / 255.0f
            byteBuffer.putFloat(normalizedPixelValue)
        }
        return byteBuffer
    }

    fun classify(image: Bitmap): Float {
        // Przeskaluj obraz do rozmiaru oczekiwanego przez model
        val resizedImage = image.resize(imageSizeX, imageSizeY)

        // Przekonwertuj obraz do ByteBuffer
        val byteBuffer = resizedImage.toByteBuffer()

        // Tablica do przechowywania wyników predykcji modelu
        val output = Array(1) { FloatArray(1) }

        // Wykonaj predykcję
        interpreter.run(byteBuffer, output)
        // Zwróć wynik klasyfikacji
        return output[0][0]
    }

}