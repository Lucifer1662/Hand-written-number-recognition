import React from 'react';
//@ts-ignore
import { Graph } from "react-d3-graph";

import NeuralNet from './NeuralNet';
import * as tf from '@tensorflow/tfjs';
//@ts-ignore
import { interpolateRgb } from 'd3-interpolate';

interface Props {
    nn: NeuralNet,
    input: tf.Tensor,
    connectionThreshold?: number,
    activationThreshold?: number,
    width?: number,
    height?: number
}

export default function NetworkGraph({ nn, input, width, height, connectionThreshold = 1.2, activationThreshold = 0.5, }: Props) {

    if(!height)
        height = 800
    if(!width)
        width = 800;

    var [az] = nn.getValues(input);

    var aBuffers = az.map(a => a.bufferSync())

    console.log(width);



    var smallNN = new NeuralNet([64, 32, 10])
    smallNN.w = nn.w.slice(1);
    smallNN.b = nn.b.slice(1);

    let bs = smallNN.b;
    let ws = smallNN.w;


    let nodes: Array<any> = [];

    let padding = 50;
    let fontSize = 30;
    let paddedHeight = height - padding * 2;
    let paddedWidth = width - padding - fontSize*7;

    var positiveColour = 'green';
    var negativeColour = 'red';



    let x = padding;
    let aIndex = 1;
    if (ws[0].shape[1]) {
        let y = padding;
        for (let i = 0; i < ws[0].shape[1]; i++) {
            nodes.push({ id: nodes.length, labelProperty: "hello", x, y, color: interpolateRgb(negativeColour, positiveColour)(aBuffers[aIndex].get(i)) });
            y += (paddedHeight / ws[0].shape[1]);
        }
    }


    bs.forEach((b) => {
        x += (paddedWidth / (bs.length+1));
        aIndex++;
        let y = padding;
        for (let i = 0; i < b.shape[0]; i++) {
            nodes.push({ id: nodes.length, x: x, y, labelProperty: () => ("hello"), color: interpolateRgb(negativeColour, positiveColour)(aBuffers[aIndex].get(i)) });
            y += (paddedHeight / b.shape[0]);
        }

    })

    var links: Array<any> = []
    var offset = 0;
    var newOffset = 0;

    ws.forEach((w, index) => {
        offset = newOffset;
        var buffer = w.bufferSync();
        if (w.shape[1])
            for (let source = 0; source < w.shape[1]; source++) {
                for (let target = 0; target < w.shape[0]; target++) {
                    if (buffer.get(target, source) < -connectionThreshold
                        && aBuffers[index + 1].get(source) > activationThreshold)
                        links.push({ color: negativeColour, source: source + offset, target: target + offset + w.shape[1] });
                    if (buffer.get(target, source) > connectionThreshold
                        && aBuffers[index + 1].get(source) > activationThreshold)
                        links.push({ color: positiveColour, source: source + offset, target: target + offset + w.shape[1] });
                }
                newOffset++;
            }
    })


    // graph payload (with minimalist structure)
    const data = {
        nodes: nodes,
        links: links
    };

    // the graph configuration, you only need to pass down properties
    // that you want to override, otherwise default ones will be used
    var nodeLabelOffset = nodes.length - bs[bs.length - 1].shape[0];

    const myConfig = {
        //@ts-ignore
        height,
        width,
        staticGraph: true,
        automaticRearrangeAfterDropNode: false,
        nodeHighlightBehavior: true,
        node: {
            color: 'lightgreen',
            size: 120,
            highlightStrokeColor: 'blue',
            fontColor: 'white',
            fontSize: 30,
            labelProperty: (node: any) => {
                if (node.id >= nodeLabelOffset)
                    return ((node.id - nodeLabelOffset) === 10 ? "NaN": node.id - nodeLabelOffset).toString() + "   " +
                        Math.floor(aBuffers[az.length - 1].get(node.id - nodeLabelOffset) * 100) + "%";
                else
                    return ""
            }
        },
        link: {
            highlightColor: 'lightblue'
        }
    };

    //@ts-ignore
    return <div  style={{ position: 'relative', width: "100%" }}>
        <Graph
            staticGraph={true}
            id="graph-id" // id is mandatory, if no id is defined rd3g will throw an error
            data={data}
            config={myConfig}


        />
    </div>;
}
