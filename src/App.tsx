import React, { useState, useRef } from 'react';
import './App.css'

//@ts-ignore
import CanvasDraw from "react-canvas-draw";
import NeuralNetwork from './NeuralNet'
import * as tf from '@tensorflow/tfjs';
import { Button } from '@material-ui/core';
import NetworkGraph from './NetworkGraph';
import useWindowSize from './useWindowSize';
var ai = require('./ai_64_32_1_junk.json')

var nn = new NeuralNetwork([28 * 28, 64, 32, 10])
nn.fromObject(ai)



function App() {
  //const [saveableCanvas, setSaveableCanvas] = useState(undefined);
  var saveableCanvasRef = useRef();
  const [number, setNumber] = useState("Not a number");
  const [input, setInput] = useState(tf.zeros([28 * 28]));
  let x = 0;
  let y = 0;
  let screenSize = useWindowSize();

  let width = 28 * 20;
  let height = 28 * 20;
  var landscape = (screenSize && screenSize.width > screenSize.height * 1);

  var graphWidth = 800;
  var graphHeight = 800;
  if(screenSize){
    graphWidth = landscape ? screenSize.width/2: screenSize.width;
    graphHeight = screenSize.height;
  }


  if (screenSize) {
    var dimension = Math.min(landscape ? screenSize.width / 3 : (landscape? screenSize.width: screenSize.width-70), screenSize.height);
    width = 28 * Math.floor((dimension) / 28);
    height = 28 * Math.floor((dimension) / 28);
  }





  const predicatNumber = async () => {
    if (saveableCanvasRef.current) {
      setNumber("...")
      //@ts-ignore
      var context = saveableCanvasRef.current.canvas.drawing.getContext('2d');
      var imgd = context.getImageData(x, y, width, height);
      var pix = imgd.data;

      const skipW = Math.floor((imgd.width * 4) / (28));
      const skipH = Math.floor((imgd.height * 4) / (28));
      let data = []


      for (let y = 0; y < imgd.height * 4; y += skipH) {
        for (let x = 0; x < imgd.width * 4; x += skipW) {
          data.push(pix[x + y * imgd.width + 3] / 255.0)
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
      setInput(xs);

      var resBuffer = await nn.oneHotHighestValue(xs).buffer()
      setNumber(resBuffer.get(0).toString())

    }
  }


  return (


    <div className="App">

      <header className="App-header">
        <div style={{ display: landscape ? 'inline' : 'flex', flexDirection:'column', width: "100%" }}>
          <div style={{ margin: 20, display: 'inline-block', width, verticalAlign:"top"}} >
          <div 
          // style={{       
          //   display: 'flex', flexDirection: 'column', justifyContent: 'flex-start', alignItems: 'center'}} 
          >
            <h2 style={{width: width}} >{number === "10" ? "Not a number" : number}</h2>
            <CanvasDraw
              key={3245}
              canvasWidth={width}
              canvasHeight={height}
              brushColor="#000000"
              brushRadius={20}
              lazyRadius={0}
              onChange={predicatNumber}

              //@ts-ignore
              ref={saveableCanvasRef}
            />
            <div style={{ margin: '20px', width }}>
              <Button
            
                variant='contained'
                onClick={() => {
                  if (saveableCanvasRef.current) {
                    //@ts-ignore
                    saveableCanvasRef.current.clear();
                    setInput(tf.zeros([28 * 28]));
                    setNumber("Not a number")
                  }
                }}
              >Clear</Button>
            </div>
          </div>
        </div>
        <div style={landscape ? { display: 'inline-block', flex: 1, width:graphWidth, height:graphHeight }
          : { display: 'block', flex: 1, height: '80vh', width: '100%' }
        }>
          <NetworkGraph nn={nn} input={input} width={graphWidth} height={graphHeight} />
        </div>
        </div>
      </header>

    </div >
  );
}

export default App;
