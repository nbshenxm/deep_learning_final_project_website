## Motivation

Market making at market-close is a popular trading strategy, as the market price during the market-close period oscillates fiercely. A market maker is someone who quotes both a buy (bid) and a sell (ask) price in a financial instrument, such as stocks, hoping to make a profit on the bid-ask spread. A key decision for a market maker is how far they would like to set the bid-ask spread. To maximize the profit, a market maker need to set the bid price and ask price as close to the market-close price range as possible. It would be ideal if we could predict the price volatility according to the then-current market condition. This project aims to address this problem by training neural network models to predict price volatility at market-close.

## Our Models

We trained three neural network models: a baseline model, a convolutional neural network (CNN) model, and a gated recurrent unit (GRU) model. 

The baseline model is a fully connected neural network with a hidden layer that contains 10 nodes. We use it as the baseline for comparison with the other two models. 

The CNN model is a convolutional neural network with a one-dimensional-convolution layer, a max-pool layer, a fully connected hidden layer of 10 nodes, and an output layer of 1 node. The convolution layer has 1 input channel, and 2 output channels, with a kernel size of 3 and stride size of 1. The max-pool layer has a kernel size of 3 and stride size of 1.

The GRU model is a recurrent neural network that contains a GRU layer with input size of 29, hidden dimension of 256, and output size of 1 followed by a ReLu activation function.

## Evaluation

We obtained S&P 500 Value Index second-by-second trading data from 09/28/2009 9:30:00 to 11/20/2020 20:00:00. We found the data from [Kibot](http://www.kibot.com/free_historical_data.aspx), a historical data vendor. We used 90% of the data for training and 10% for testing. For each epoch, we keep track of the training loss and the accuracy of the model.

We measured the accuracy of the models by calculating the symmetric mean absolute percentage error (sMAPE). The formula for calculating sMAPE is 

![smape](/images/smape.png)

where F_t is the forecast value and A_t is the actual value. We chose the sMAPE method over the classic MAPE method because sMAPE is resistant to outliers. The graphs below show the evalution of our three models.

baseline model:

![baseline](/images/baseline.png)

CNN model:

![cnn](/images/cnn.png)

GRU model:

![GRU_1](/images/GRU_epoch100_lr00001_hidden256_nlayers2.png)

The table below shows the evaluation of GRU models with various parameters.

| Epoch | Learning rate | Hidden layer dimension | Number of layers | Percentage error |
| --- | ----------- | ----------- | ----------- | ----------- |
| 100 | 0.001 | 128 | 2 | 16.4% |
| 500 | 0.001 | 128 | 2 | 19.2% |
| 1000 | 0.001 | 128 | 2 | 17.4% |
| 100 | 0.01 | 128 | 2 | 17.2% |
| 100 | 0.0001 | 128 | 2 | 12.9% |
| 100 | 0.0001 | 64 | 2 | 16.9% |
| 100 | 0.0001 | 256 | 2 | 11.4% |
| 100 | 0.0001 | 512 | 2 | 17.4% |
| 100 | 0.0001 | 256 | 3 | 18.6% |


## Discussion

All 3 models could accurately forecast the price volatility, with the GRU model being the most accurate model, followed by the baseline model, then followed by the CNN model. We expected the GRU model to be the best performing model because this model is designed for processing time-sequence data. We did not expect the CNN model to perform the poorest. One explanation could be that our input has a size of 29, which might be too small to use for a CNN model. We might have over generalized the features during convolution and pooling.