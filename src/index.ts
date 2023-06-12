import 'materialize-css/dist/js/materialize.min.js';
import './styles.css';

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

// globals
let model: tf.Sequential|tf.LayersModel;
let normalisedFeature: {min: any; max: any; tensor: any};
let normalisedLabel: {min: any; max: any; tensor: any};
let trainingFeatures: any;
let testingFeatures: any;
let trainingLabels: any;
let testingLabels: any;
const localStorageID = 'house-price-regression';
const points: {x: any; y: any}[] = [];

// utils
const plot = (points: any[], featureName: string, predictedPoints: null|{
  x: number;
  y: number
}[] = null) => {
  const values = [points];
  const series = ['original'];

  if (Array.isArray(predictedPoints)) {
    values.push(predictedPoints);
    series.push('prediceted');
  }

  tfvis.render.scatterplot(
      {name: `${featureName} vs House Price`}, {values, series}, {
        xLabel: featureName,
        yLabel: 'Price',
      });
};
const plotPrediction = async () => {
  const [xs, ys] = tf.tidy(() => {
    const normalisedXs =
        tf.linspace(0, 1, 100); // range of 100 points within 0 and 1
    const normalisedYs = model.predict(
        normalisedXs.reshape([100, 1])); // predict expects 2d tensor

    const xs =
        denormalise(normalisedXs, normalisedFeature.min, normalisedFeature.max);
    const ys = denormalise(
        normalisedYs as tf.Tensor<tf.Rank.R1>, normalisedLabel.min,
        normalisedLabel.max);

    return [xs.dataSync(), ys.dataSync()];
  });

  console.log('XS', xs);
  console.log('YS', ys);

  const predictedPoints = Array.from(xs).map((x, i) => ({x, y: ys[i]}));

  await plot(points, 'Square feet', predictedPoints);
};
const normalise =
    (tensor: tf.Tensor2D|tf.Tensor1D, previousMin = null,
        previousMax = null) => {
      const max = previousMax ?? tensor.max();
      const min = previousMin ?? tensor.min();
      const normalisedTensor = tensor.sub(min).div(max.sub(min));

      return {tensor: normalisedTensor, min, max};
    };
const denormalise = (tensor: tf.Tensor1D, min: any, max: string) => {
  return tensor.mul(tf.sub(max, min)).add(min);
};
const trainModel = async (
    model: tf.Sequential, trainingFeatureTensor: any,
    trainingLabelTensor: any) => {
  const {onEpochEnd} = tfvis.show.fitCallbacks(
      {
        name: 'Training Performance',
      },
      ['loss']);

  return model.fit(trainingFeatureTensor, trainingLabelTensor, {
    batchSize: 32,
    epochs: 20,
    callbacks: {
      onEpochBegin: async () => {
        await plotPrediction();
        // update layer summary showing current values of the weights
        const layer = model.getLayer(undefined, 0); // 1st and only layer
        tfvis.show.layer({name: 'Layer'}, layer);
      },
      onEpochEnd,
    } as any,
  });
};

const createModel = () => {
  model = tf.sequential();
  (model as tf.Sequential).add(tf.layers.dense({
    units: 1, // single node
    useBias: true,
    activation: 'linear', // no threshold
    inputDim: 1,
  }));

  return model;
};

const run = async () => {
  await tf.ready();
  const houseSalesDataset =
      tf.data.csv('http://127.0.0.1:3000/kc_house_data.csv');
  const pointsDataset = houseSalesDataset.map((record: any) => ({
    x: record.sqft_living,
    y: record.price,
  }));

  await pointsDataset.forEachAsync((point: {x: any; y: any}) => {
    points.push(point);
  });

  if (points.length % 2 !== 0) {
    points.pop();
  }

  const featureValues = points.map((p) => p.x);
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);
  const labelValues = points.map((p) => p.y);
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

  normalisedFeature = normalise(featureTensor);
  normalisedLabel = normalise(labelTensor);

  const trainingSize = Math.round(normalisedFeature.tensor.shape[0] * 0.8);
  const testingSize = Math.round(normalisedFeature.tensor.shape[0] * 0.2);

  [trainingFeatures, testingFeatures] = tf.split(normalisedFeature.tensor, [
    trainingSize,
    testingSize,
  ]);
  [trainingLabels, testingLabels] = tf.split(normalisedLabel.tensor, [
    trainingSize,
    testingSize,
  ]);

  // update state and enable train button
  document.getElementById('model-status')!.innerHTML = 'No model trained';
  document.getElementById('train-button')!.removeAttribute('disabled');
  document.getElementById('load-button')!.removeAttribute('disabled');
};

export const predict = async () => {
  const predictionInput = parseInt(
      (document.getElementById('prediction-input')! as HTMLInputElement).value);
  if (isNaN(predictionInput)) {
    alert('Please enter a valid number');
  } else {
    tf.tidy(() => {
      const inputTensor = tf.tensor1d([predictionInput]);
      const normalisedInput =
          normalise(inputTensor, normalisedFeature.min, normalisedFeature.max);
      const normalisedOutputTensor = model.predict(normalisedInput.tensor);
      const outputTensor = denormalise(
          normalisedOutputTensor as tf.Tensor<tf.Rank.R1>, normalisedLabel.min,
          normalisedLabel.max);
      const outputValue = outputTensor.dataSync()[0];
      const outputValueRounded = Math.round(outputValue / 1000) * 1000;
      document.getElementById('prediction-output')!.innerHTML =
          `$${outputValueRounded}`;
    });
  }
};

export const load = async () => {
  const storageKey = `localstorage://${localStorageID}`;
  const models = await tf.io.listModels();
  const modelInfo = models[storageKey];
  if (modelInfo) {
    model = (await tf.loadLayersModel(storageKey)) as tf.LayersModel;

    // show model on visor
    const layer = model.getLayer(undefined, 0); // 1st and only layer
    tfvis.show.modelSummary({name: `Model Summary`, tab: `Model`}, model);
    tfvis.show.layer({name: 'Layer'}, layer);

    await plotPrediction();

    // update statuses
    document.getElementById('model-status')!.innerHTML =
        `Trained (saved ${modelInfo.dateSaved})`;
    document.getElementById('test-button')!.removeAttribute('disabled');
    document.getElementById('predict-button')!.removeAttribute('disabled');
  } else {
    alert('Could not load: no model found');
  }
};

export const save = async () => {
  const saveResults = await model.save(`localstorage://${localStorageID}`);
  document.getElementById('model-status')!.innerHTML =
      `Trained (saved ${saveResults.modelArtifactsInfo.dateSaved})`;
};

export const test = async () => {
  const lossTensor = model.evaluate(testingFeatures, testingLabels);
  const loss = ((await lossTensor) as tf.Scalar)
      .dataSync(); // data sync returns numeric value

  document.getElementById('testing-status')!.innerHTML =
      `Testing set loss: ${Number(loss).toPrecision(5)}`;
};

export const train = async () => {
  // disable all buttons
  ['train', 'test', 'load', 'predict', 'save'].forEach((id) => {
    document.getElementById(`${id}-button`)
        ?.setAttribute('disabled', 'disabled');
  });
  document.getElementById('model-status')!.innerHTML = 'Training...';

  model = createModel();

  model.summary();
  const layer = model.getLayer(undefined, 0); // 1st and only layer
  tfvis.show.modelSummary({name: `Model Summary`, tab: `Model`}, model);
  tfvis.show.layer({name: 'Layer'}, layer);

  const optimizer = tf.train.sgd(0.1);

  model.compile({
    loss: 'meanSquaredError',
    optimizer,
  });

  await plotPrediction();

  const result = await trainModel(
      model as tf.Sequential, trainingFeatures, trainingLabels);
  const trainingLoss = result.history.loss.pop();

  document.getElementById('model-status')!.innerHTML = 'Trained (unsaved) \n' +
      `Loss ${Number(trainingLoss).toPrecision(5)}\n`;

  document.getElementById('test-button')?.removeAttribute('disabled');
  document.getElementById('save-button')?.removeAttribute('disabled');
};

export const toggleVisor = async () => {
  tfvis.visor().toggle();
};

tf.tidy(() => {
  run();
});
