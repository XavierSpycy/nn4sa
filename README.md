<h1><strong>nn4sa: Neural Networks for Sentiment Analysis</strong></h1>

![Maintain](https://img.shields.io/badge/maintained-yes-green) 
![License](https://img.shields.io/github/license/XavierSpycy/nn4sa) 
![Contribution](https://img.shields.io/badge/contributions-welcome-brightgreen)

<h2>Data Description</h2>
<ul>
    <li>
        Shape: 300,000 &times; 6 (Note: Only 2 columns are utilized.)
    </li>
    <li>
        <p><code>text</code>: Tweets;</p>
        <p><code>sentiment</code>: Positive / Negative.</p>
    </li>
</ul>

<h2>Train an RNN</h2>
<ul>
    <li>
        <p><u>Description</u>:</p>
        <p>Recurrent Neural Networks (RNNs) were preeminent models several years ago. Despite the advent of powerful Transformers, RNNs remain popular due to their computational efficiency.</p>
    </li>
    <li>
        <p><u>Usage</u>:</p>
        <p><code>$ python3 scripts/train.py --method rnn # Options: lstm, gru</code></p>
    </li>
</ul>

<h2>Fine-tune BERT</h2>
<ul>
    <li>
        <p><u>Description</u>:</p>
        <p>BERT has been a milestone in DL and AI, revolutionizing the NLP paradigm. It serves as a robust feature extractor. Despite the rise of LLMs, BERT continues to excel in numerous downstream tasks, including sentiment analysis.</p>
    </li>
    <li>
        <p><u>Usage</u>:</p>
        <p><code>$ python3 scripts/train.py --method bert --bert_model_path /path/to/your/bert/model</code></p>
    </li>
</ul>

<h2>Apply a State-of-the-Art Embedding Model</h2>
<ul>
    <li>
        <p><u>Description</u>:</p>
        <p>With advances in LLMs and RAG, various embedding models have emerged. We recommend <code>Salesforce/SFR-Embedding-Mistral</code> as an example. Typically, most embedding models are used similarly.</p>
    </li>
    <li>
        <u>Usage</u>:
        <ol>
            <li>
                <p>Step 1: Extract Embeddings (This may take several hours, depending on the model size.)</p>
                <p><code>$ python3 scripts/embed.py --cache_dir /path/to/your/preferred/location</code></p>
            </li>
            <li>
                <p>Step 2: Train a Classification Head</p>
                <p><code>$ python3 scripts/train.py --method embed --embed_path /path/to/your/embed/model</code></p>
            </li>
        </ol>
    </li>
</ul>

<h2>Results</h2>

<div align="center">
    <table>
        <tr>
            <td align="center">Method</td>
            <td align="center">Train Accuracy</td>
            <td align="center">Test Accuracy</td>
            <td align="center">Number of Epochs</td>
            <td align="center">Efficiency Bottleneck</td>
        </tr>
        <tr>
            <td align="center">RNN from Scratch (LSTM)</td>
            <td align="center">77.98%</td>
            <td align="center">76.79%</td>
            <td align="center" rowspan="2">3</td>
            <td align="center">Preprocessing</td>
        </tr>
        <tr>
            <td align="center">Fine-tuned BERT</td>
            <td align="center">76.04%</td>
            <td align="center">78.46%</td>
            <td align="center">N/A</td>
        </tr>
        <tr>
            <td align="center">SOTA Embedder</td>
            <td align="center">85.98%</td>
            <td align="center">85.71%</td>
            <td align="center">4</td>
            <td align="center">Embedding</td>
        </tr>
    </table>
</div>

> [!NOTE]
> RNNs typically infer quickly. However, training from scratch means building a tokenizer from scratch as well, which in our case is straightforward but more time-consuming than the actual training process. Saving the tokenizer vocabulary significantly enhances efficiency for subsequent uses.
> The training process for a classification head using state-of-the-art embedding models is efficient when split into `embedding extraction` and `classification head training` from an engineering perspective. Nevertheless, `embedding extraction` is quite time-consuming. In our case, processing 300,000 samples took about 24 hours on 2 RTX 4090 GPUs with a batch size of 1.