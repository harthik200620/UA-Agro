// AudioWorklet processor: forwards mono Float32 frames to the main thread.
// The main thread runs VAD, downsamples to 16 kHz, and encodes WAV.
class PCMProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      // copy so the buffer isn't recycled under us
      this.port.postMessage(input[0].slice(0));
    }
    return true; // keep the node alive
  }
}
registerProcessor('pcm-processor', PCMProcessor);
