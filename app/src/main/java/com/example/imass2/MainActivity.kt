package com.example.imass2

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.imass2.ui.theme.Imass2Theme
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.model.Model
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.io.FileInputStream
//import java.io.FileChannel
import android.app.Activity
import android.app.AlertDialog
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Card
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import com.example.imass2.ml.MobilenetV110224Quant

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            Imass2Theme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    ImageClassificationScreen(context = this)
                }
            }
        }
    }
}

@Composable
fun ImageClassificationScreen(context: Context) {
    var classificationResult by remember { mutableStateOf("") }
    var selectedImageBitmap by remember { mutableStateOf<Bitmap?>(null) }

    val galleryLauncher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let {
            val bitmap = BitmapFactory.decodeStream(context.contentResolver.openInputStream(it))
            selectedImageBitmap = bitmap
            val result = classifyBitmap(context, bitmap)
            classificationResult = "Classification result: $result"
        }
    }

    Column(
        modifier = Modifier.fillMaxSize(),
    ) {
        selectedImageBitmap?.let { bitmap ->
            Card(
                modifier = Modifier.padding(16.dp)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = "Selected Image",
                        modifier = Modifier.fillMaxWidth()
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = classificationResult,
                        modifier = Modifier.padding(8.dp)
                    )
                }
            }
        }
        Button(
            onClick = {
                galleryLauncher.launch("image/*")
            },
            modifier = Modifier.padding(16.dp)
                .align(Alignment.CenterHorizontally)
        ) {
            Text("Select Image from Gallery")
        }
    }
}

private fun classifyBitmap(context: Context, bitmap: Bitmap): String {
    // Load the TensorFlow Lite model
    val model = MobilenetV110224Quant.newInstance(context)

    // Resize the bitmap to match the input size of the model
    val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

    // Convert the resized bitmap to a TensorImage
    val inputImage = TensorImage(DataType.UINT8)
    inputImage.load(resizedBitmap)

    // Create an input TensorBuffer
    val inputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)

    // Copy the data from the input TensorImage to the input TensorBuffer
    val byteBuffer = inputImage.buffer
    byteBuffer.order(ByteOrder.nativeOrder())
    inputBuffer.loadBuffer(byteBuffer)

    // Run model inference
    val outputs = model.process(inputBuffer)

    // Get the output TensorBuffer
    val outputBuffer = outputs.outputFeature0AsTensorBuffer

    // Get the label for the highest probability output
    val labels = loadLabels(context, "labels.txt")
    val labeledProbability = TensorLabel(labels, outputBuffer).mapWithFloatValue
    val maxValue = labeledProbability.values.maxOrNull()
    val label = labeledProbability.filterValues { it == maxValue }.keys.first()

    // Release the model resources
    model.close()

    return label
}

private fun loadLabels(context: Context, labelsFilename: String): List<String> {
    return context.assets.open(labelsFilename).bufferedReader().readLines()
}
