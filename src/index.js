import '../node_modules/materialize-css/dist/js/materialize.min.js';
import './styles.css';

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

// utils
const plot = (points, featureName) => {
  tfvis.render.scatterplot(
      {name: `${featureName} vs House Price`},
      {values: [points], series: ['original']},
      {
        xLabel: featureName,
        yLabel: 'Price',
      },
  );
};
const normalise = (tensor) => {
  const max = tensor.max();
  const min = tensor.min();
  const normalisedTensor = tensor.sub(min).div(max.sub(min));

  return {tensor: normalisedTensor, min, max};
};
const denormalise = (tensor, min, max) => {
  return tensor.mul(max.sub(min)).add(min);
};
const trainModel =
    async (model, trainingFeatureTensor, trainingLabelTensor) => {
      const {onBatchEnd, onEpochEnd} = tfvis.show.fitCallbacks(
          {
            name: 'Training Performance',
          },
          ['loss']);

      return model.fit(trainingFeatureTensor, trainingLabelTensor, {
        batchSize: 32,
        epochs: 20,
        callbacks: {
          onEpochEnd,
        },
      });
    };
const createModel = () => {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    units: 1, // single node
    useBias: true,
    activation: 'linear', // no threshold
    inputDim: 1,
  }));

  return model;
};

//
const run = async () => {
  await tf.ready();
  const houseSalesDataset =
      tf.data.csv('http://127.0.0.1:3000/kc_house_data.csv');
  const pointsDataset = houseSalesDataset.map(
      (record) => ({x: record.sqft_living, y: record.price}));
  const points = [];

  await pointsDataset.forEachAsync(
      (point) => {
        points.push(point);
      },
  );
  console.log('points', points);

  if (points.length % 2 !== 0) {
    points.pop();
  }

  const featureValues = points.map((p) => p.x);
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);
  const labelValues = points.map((p) => p.y);
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);
  plot(points, 'Square Feet');

  const normalisedFeature = normalise(featureTensor);
  const normalisedLabel = normalise(labelTensor);

  const trainingSize = Math.round(normalisedFeature.tensor.shape[0] * 0.8);
  const testingSize = Math.round(normalisedFeature.tensor.shape[0] * 0.2);

  const [trainingFeatures, testingFeatures] =
      tf.split(normalisedFeature.tensor, [trainingSize, testingSize]);
  const [trainingLabels, testingLabels] =
      tf.split(normalisedLabel.tensor, [trainingSize, testingSize]);

  const model = createModel();

  model.summary();
  const layer = model.getLayer(undefined, 0); // 1st and only layer

  tfvis.show.modelSummary({name: `Model Summary`, tab: `Model`}, model);
  tfvis.show.layer({name: 'Layer'}, layer);

  const optimizer = tf.train.sgd(0.1);

  model.compile({
    loss: 'meanSquaredError',
    optimizer,
  });

  const result = await trainModel(model, trainingFeatures, trainingLabels);
  const trainingLoss = result.history.loss.pop();

  const lossTensor = model.evaluate(testingFeatures, testingLabels);
  const loss = await lossTensor.dataSync(); // data sync returns numeric value
};

export const predict = async () => {
  alert('Not yet implemented');
};

export const load = async () => {
  alert('Not yet implemented');
};

export const save = async () => {
  alert('Not yet implemented');
};

export const test = async () => {
  alert('Not yet implemented');
};

export const train = async () => {
  alert('Not yet implemented');
};

export const toggleVisor = async () => {
  tfvis.visor().toggle();
};

tf.tidy(() => {
  run();
});
