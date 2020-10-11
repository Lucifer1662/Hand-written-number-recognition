import React, { useState } from 'react';
import './App.css';
//@ts-ignore
import CanvasDraw from "react-canvas-draw";
import NeuralNetwork from './NeuralNet'
import * as tf from '@tensorflow/tfjs';
import {Button} from '@material-ui/core';
var ai = require('./ai_64_32_1_junk.json')

var nn = new NeuralNetwork([28*28, 64, 32, 10])
nn.fromObject(ai) 

function App() {
  const [saveableCanvas, setSaveableCanvas] = useState(undefined);
  const [number, setNumber] = useState("");
  let x = 0;
  let y = 0;
  let width = 28*20;
  let height = 28*20;




  const predicatNumber = async () => {
    if (saveableCanvas) {
      setNumber("...")
      //@ts-ignore
      var context = saveableCanvas.canvas.drawing.getContext('2d');
      var imgd = context.getImageData(x, y, width, height);
      var pix = imgd.data;

      const skipW = Math.floor((imgd.width * 4) / (28));
      const skipH = Math.floor((imgd.height * 4) / (28));
      let data = []
     
      
      for (let y = 0; y < imgd.height*4; y+=skipH) {
        for (let x = 0; x < imgd.width*4; x+=skipW) {
          data.push(pix[x + y*imgd.width + 3]/255.0)
        }
      }
  

      // var i =0;
      // var s = ""
      // for (let y = 0; y < 28; y++) {
      //   for (let x = 0; x < 28; x++) {
      //     if(data[i] < 0.5)
      //       s += 0
      //     else
      //       s += 1

      //     i++;          
      //   }     
      //   s += '\n'
            
      // }

      var xs = tf.tensor(data, [28, 28]);
     
      xs = xs.flatten();
      
      var resBuffer = await nn.oneHotHighestValue(xs).buffer()
      setNumber(resBuffer.get(0).toString())

    }
  }


  return (
    <div className="App">
      <header className="App-header">
        <h4>{number}</h4>
        <CanvasDraw
          canvasWidth={width}
          canvasHeight={height}
          brushColor= "#000000"
          brushRadius={20}
          lazyRadius={0}
          onChange={predicatNumber}

          //@ts-ignore
          ref={(canvasDraw: any) => (setSaveableCanvas(canvasDraw))} />
          <div style={{margin:'20px'}}>
        <Button
        
        variant='contained'
          onClick={() => {
            if (saveableCanvas) {
              //@ts-ignore
              saveableCanvas.clear();
            }
          }}
        >Clear</Button>
        </div>
      </header>
    </div>
  );
}

export default App;
