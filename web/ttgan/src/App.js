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
        </header>
      </div>
    );
  }
}

export default App;
