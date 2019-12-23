import React from 'react';
import logo from './logo.svg';
import './App.css';
import GANControls from './GANControls';
import GANImage from './GANImage';

class App extends React.Component {
  render() {
    return (
      <div className="App">
        <header className="App-header">
          <p>
            Explore tartan space
          </p>
          <GANImage modelSrc="ttgan.onnx" />
          <GANControls />
          <a
            className="App-link"
            href="https://awentzonline.github.io/tartangan"
            target="_blank"
            rel="noopener noreferrer"
          >
            TartanGAN
          </a>
        </header>
      </div>
    );
  }
}

export default App;
