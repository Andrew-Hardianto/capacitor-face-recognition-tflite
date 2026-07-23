import { FaceRecognition } from 'capacitor-face-recognition-tflite';

window.testEcho = () => {
    const inputValue = document.getElementById("echoInput").value;
    FaceRecognition.echo({ value: inputValue })
}
