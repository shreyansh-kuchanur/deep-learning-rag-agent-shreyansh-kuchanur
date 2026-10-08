# Long Short-Term Memory (LSTM) Networks

## What is an LSTM?

A Long Short-Term Memory network (LSTM) is a special type of recurrent neural network (RNN) designed to learn long-range dependencies in sequential data. It was introduced by Hochreiter and Schmidhuber in 1997. LSTMs are used when the order of the data matters, such as text, speech, and time series.

## Why do we need LSTMs?

Standard RNNs suffer from the vanishing gradient problem. During backpropagation through time, gradients are multiplied many times by small values, so they shrink towards zero. As a result, a plain RNN struggles to remember information from many time steps earlier. LSTMs solve this problem by adding a separate memory path, called the cell state, and gates that control what is stored, forgotten, and output.

## The cell state

The cell state is the long-term memory of the LSTM. It runs along the whole sequence with only small, controlled changes, which allows gradients to flow backwards over many time steps without vanishing. The gates decide how the cell state is updated at each time step.

## The forget gate

The forget gate decides which information should be removed from the cell state. It looks at the previous hidden state and the current input and passes them through a sigmoid function. The sigmoid outputs a value between 0 and 1 for each number in the cell state: 0 means "completely forget this" and 1 means "completely keep this". For example, in a language model, the forget gate can drop the gender of an old subject when a new subject appears in the sentence.

## The input gate

The input gate decides which new information should be written to the cell state. It has two parts: a sigmoid layer that chooses which values to update, and a tanh layer that creates candidate values to add. The two are multiplied together, and the result is added to the cell state after the forget gate has removed old information.

## The output gate

The output gate decides what the LSTM outputs as its hidden state at the current time step. A sigmoid layer selects which parts of the cell state to expose. The cell state is passed through tanh to squash its values between -1 and 1, and then multiplied by the output of the sigmoid. The resulting hidden state is used for predictions and is passed to the next time step.

## Hidden state versus cell state

The hidden state is the short-term output of the LSTM at each time step and is what the next layer sees. The cell state is the internal long-term memory that is carried forward and is mostly hidden from the rest of the network. Having both lets the LSTM separate what it remembers from what it reports.

## Common use cases

LSTMs are commonly used for:

- Language modelling and text generation
- Machine translation, often inside encoder-decoder (Seq2Seq) models
- Speech recognition
- Time series forecasting, such as stock prices or energy demand
- Sentiment analysis of sentences and reviews

## Limitations

LSTMs process sequences one step at a time, so they cannot be parallelised easily and are slow to train on long sequences. They also have many parameters compared with simpler RNNs. For many modern language tasks, Transformers have replaced LSTMs because self-attention can be computed in parallel and handles very long contexts better. A simpler alternative is the Gated Recurrent Unit (GRU), which merges the forget and input gates into a single update gate.

## Interview summary

An LSTM is an RNN with a cell state and three gates. The forget gate removes old information, the input gate adds new information, and the output gate controls the hidden state. This gated design reduces the vanishing gradient problem and lets the network remember information over long sequences.
