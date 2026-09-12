<h1 align="center">📈 Reliance Stock Price Forecasting using LSTM</h1>

<p align="center">
A time-series deep learning project that forecasts Reliance Industries' next-day closing stock price using a stacked LSTM network, with every stage — scaling, sequencing, gate math, and evaluation — worked through manually alongside the Keras implementation.
</p>

<hr>

<h2>📌 Project Overview</h2>

<p>
This project implements an <b>LSTM (Long Short-Term Memory) network</b> to forecast the next-day closing price of Reliance Industries stock from historical OHLCV data.
Rather than treating the model as a black box, the pipeline walks through each transformation step explicitly — manual Min-Max scaling, manual sequence construction, and even a hand-computed LSTM gate pass — before handing the same data to a Keras-trained model.
</p>

<p>
The goal is a transparent, end-to-end sequence-modeling pipeline: understanding exactly how raw daily prices become normalized sequences, how an LSTM's gates process them internally, and how scaled predictions are converted back into real rupee values and evaluated.
</p>

<ul>
<li>Understanding time-series windowing (sliding sequences) for supervised learning</li>
<li>Working through LSTM internals — forget, input, candidate, and output gates — by hand</li>
<li>Building and training a stacked LSTM with dropout regularization in Keras</li>
<li>Evaluating forecasts with MAE, MSE, and RMSE, and generating a live next-day prediction</li>
</ul>

<hr>

<h2>⚙️ Features</h2>

<ul>
<li>Loads and inspects historical daily OHLCV data for Reliance Industries</li>
<li>Manual Min-Max scaling walkthrough alongside <code>MinMaxScaler</code>, with formula and worked examples</li>
<li>Sliding-window sequence generation (10 previous days → next day's price)</li>
<li>Reshaping into the 3D input format LSTM layers expect (samples, time steps, features)</li>
<li>Step-by-step manual LSTM gate calculation demo (forget, input, candidate, output, hidden state)</li>
<li>Stacked LSTM architecture (2 LSTM layers + dropout + dense layers) trained with early stopping</li>
<li>Inverse scaling of predictions back to original rupee values, with manual verification</li>
<li>Manual and library-based MAE / MSE / RMSE evaluation</li>
<li>Actual-vs-predicted and training-loss plots saved as PNGs</li>
<li>Live next-day closing price prediction from the most recent 10 trading days</li>
<li>Trained model and prediction outputs saved to disk</li>
</ul>

<h3>Pipeline Stages</h3>

<ul>
<li>Dataset Loading &amp; Inspection</li>
<li>Date Conversion &amp; Sorting</li>
<li>Close Price Extraction</li>
<li>Min-Max Scaling (manual + library)</li>
<li>Sequence Creation (sliding window)</li>
<li>Reshape for LSTM Input</li>
<li>LSTM Gate Mechanics Walkthrough</li>
<li>Train/Test Split</li>
<li>Model Building &amp; Training</li>
<li>Prediction &amp; Inverse Scaling</li>
<li>Evaluation (MAE / MSE / RMSE)</li>
<li>Visualization</li>
<li>Next-Day Prediction</li>
<li>Model &amp; Output Persistence</li>
</ul>

<hr>

<h2>🛠️ Tech Stack</h2>

<ul>
<li><b>Language:</b> Python</li>
<li><b>Deep Learning:</b> TensorFlow / Keras</li>
<li><b>Data Handling:</b> Pandas, NumPy</li>
<li><b>Preprocessing:</b> Scikit-learn (MinMaxScaler)</li>
<li><b>Visualization:</b> Matplotlib</li>
</ul>

<h3>Concepts Used</h3>

<ul>
<li>Recurrent Neural Networks &amp; LSTM Architecture</li>
<li>Time-Series Windowing / Sequence-to-One Forecasting</li>
<li>Feature Scaling &amp; Inverse Transformation</li>
<li>Dropout Regularization</li>
<li>Regression Evaluation Metrics (MAE, MSE, RMSE)</li>
</ul>

<hr>

<h2>📂 Project Structure</h2>

<pre>
Reliance_Stock_Price_Forecasting_using_LSTM/
│
├── Reliance_LSTM_Time_Series.py              # Main script (main entry point)
├── reliance_stock_data.csv                   # Historical OHLCV dataset
├── reliance_lstm_detailed_model.h5           # Saved trained model
├── reliance_prediction_output.csv            # Saved prediction results
├── reliance_actual_vs_predicted_detailed.png # Actual vs predicted plot
├── reliance_training_loss_detailed.png       # Training/validation loss plot
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
</pre>

<hr>

<h2>🚀 Installation</h2>

<h3>1️⃣ Clone the Repository</h3>
<pre>
git clone https://github.com/UmeshBhabad/Reliance_Stock_Price_Forecasting_using_LSTM.git
cd Reliance_Stock_Price_Forecasting_using_LSTM
</pre>

<h3>2️⃣ Create a Virtual Environment (Recommended)</h3>
<pre>
python -m venv myvenv
myvenv\Scripts\activate      # Windows
source myvenv/bin/activate   # macOS/Linux
</pre>

<h3>3️⃣ Install Dependencies</h3>
<pre>
pip install -r requirements.txt
</pre>

<hr>

<h2>▶️ Usage</h2>

<pre>
python Reliance_LSTM_Time_Series.py
</pre>

<p>
The script runs the full pipeline end-to-end in the terminal — printing each processing step, training progress, evaluation metrics, and the next-day price prediction — and saves the trained model, prediction CSV, and plots to disk.
</p>

<hr>

<h2>🖥️ How It Works</h2>

<ol>
<li>Historical daily OHLCV data for Reliance is loaded and inspected</li>
<li>The Date column is converted to datetime and the data sorted chronologically</li>
<li>Closing prices are extracted as the forecasting target</li>
<li>Prices are Min-Max scaled to the 0–1 range (with a manual calculation shown alongside)</li>
<li>Overlapping 10-day input sequences are built, each paired with the next day's price as the label</li>
<li>Sequences are reshaped into the 3D (samples, time steps, features) format LSTM layers require</li>
<li>A stacked 2-layer LSTM with dropout is built, compiled with Adam and MSE loss, and trained</li>
<li>The model predicts scaled closing prices on the held-out test set</li>
<li>Predictions are inverse-scaled back to real rupee values and compared against actuals</li>
<li>MAE, MSE, and RMSE are computed, and actual-vs-predicted / loss curves are plotted</li>
<li>The last 10 known trading days are fed back into the model to forecast the next day's close</li>
</ol>

<hr>

<h2>📊 Model Performance</h2>

<p>
Trained on 130 daily records (Jan – Jun 2024), split into 120 sliding-window sequences (10-day lookback), with an 80/20 train-test split. Training stopped early at epoch 12/60 once validation loss stopped improving.
</p>

<pre>
Final Training Loss   : 0.0240 (scaled)
Final Validation Loss : 0.0654 (scaled)

Mean Absolute Error (MAE) : ₹16.10
Mean Squared Error (MSE)  : 403.16
Root Mean Squared Error   : ₹20.08
</pre>

<p>
<i>Note: trained on a single stock over a 6-month window (130 trading days), so these figures reflect model behavior on a small, single-ticker dataset rather than general market performance. Errors of ₹15–35 are visible on days with sharper price swings — a natural limitation of a 10-day lookback window, and a good candidate for the multivariate and longer-context improvements noted below.</i>
</p>

<hr>

<h2>🧪 Example Interaction</h2>

<pre>
============================================================
Last 10 Trading Days (Close Price)
============================================================
Day 01: 2418.80   Day 06: 2441.98
Day 02: 2435.33   Day 07: 2441.74
Day 03: 2446.02   Day 08: 2433.85
Day 04: 2443.27   Day 09: 2420.87
Day 05: 2427.89   Day 10: 2423.36

Next Day Prediction:
Predicted Scaled Value        : 0.32282
Predicted Original Close Price: ₹2419.61
</pre>

<hr>

<h2>🧠 Design Highlights</h2>

<ul>
<li><b>Transparent Math</b> — scaling, sequencing, and LSTM gate calculations are worked manually alongside the library calls, not just called as black-box functions</li>
<li><b>Sliding-Window Sequencing</b> — 10-day lookback windows turn a univariate price series into a supervised learning problem</li>
<li><b>Regularized Architecture</b> — stacked LSTM layers with dropout reduce overfitting on a relatively small dataset</li>
<li><b>Full Inverse-Scaling Loop</b> — predictions are always converted back to real rupee values before evaluation, with a manual check against the formula</li>
<li><b>Reproducible Persistence</b> — trained model, prediction outputs, and plots are all saved for later inspection or reuse</li>
</ul>

<hr>

<h2>🔮 Future Enhancements</h2>

<ul>
<li>Multivariate input (Open, High, Low, Volume) instead of Close price alone</li>
<li>Longer historical window and larger dataset for more robust generalization</li>
<li>Hyperparameter tuning (time steps, units, dropout rate) via grid/random search</li>
<li>Walk-forward validation instead of a single train/test split</li>
<li>Attention-based or Transformer-based sequence models for comparison</li>
<li>Live data ingestion via a stock market API for real-time forecasting</li>
</ul>

<hr>

<h2>👨‍💻 Author</h2>

<p>
<b>Umesh Shivaji Bhabad</b><br>
📫 umeshbhabad9@gmail.com
</p>

<hr>

<h2>⭐ Support</h2>

<p>If you find this project useful, consider giving it a ⭐ on GitHub!</p>