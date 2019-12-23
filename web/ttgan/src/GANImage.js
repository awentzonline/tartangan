import React from 'react';
import { Tensor, InferenceSession } from "onnxjs";

class GANImage extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      imgWidth: 128,
      imgHeight: 128,
    }
  }

  async componentDidMount() {
    this.session = new InferenceSession();
    const url = this.props.modelSrc
    await this.session.loadModel(url);
    await this.updateImage();
  }

  async updateImage() {
    const latentDims = 256;
    const batchSize = 1;
    const x = new Float32Array(batchSize * latentDims).map(() => Math.random() * 2 - 1)
    const tensorX = new Tensor(x, 'float32', [batchSize, latentDims]);
    const outputMap = await this.session.run([tensorX]);
    const imgBatch = outputMap.values().next().value;
    const width = imgBatch.dims[3]
    const height = imgBatch.dims[2]
    this.setState({
      imgWidth: width, imgHeight: height
    });
    let scaledPixels = imgBatch.data.map(x => x * 255)
    const bytes = new Uint8ClampedArray(scaledPixels);
    // HACK: not sure how to add a dimension to the ONNX tensor so
    // I'm doing this to add an alpha channel instead of
    // simply dumping the bytes into ImageData into canvas context.
    // const imageData = new ImageData(bytes, width, height);
    // ctx.putImageData(imageData, 0, 0);
    const ctx = this.refs.canvas.getContext('2d');
    ctx.fillStyle = '#ffffff'; // implicit alpha of 1
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    const destData = ctx.getImageData(0, 0, width, height)
    var dest = destData.data
    var n = 4 * width * height;
    var s = 0, d = 0;
    const src = bytes;
    while (d < n) {
        dest[d++] = src[s++];
        dest[d++] = src[s++];
        dest[d++] = src[s++];
        d++;    // skip the alpha byte
    }
    ctx.putImageData(destData, 0, 0);
  }

  render() {
    return <div>
      <canvas ref="canvas"
        width={this.state.imgWidth} height={this.state.imgHeight}
        style={{width:256, height:256}}
      />
    </div>
  }
}

GANImage.defaultProps = {
  modelSrc: 'ttgan.onnx',
  imgWidth: 128,
  imgHeight: 128
}

export default GANImage;
