import venumpy
import numpy as np
import pandas as pd
from venumMLlib.venum_tools import encrypt_array
from venumMLlib.linear_models.regression.linear_regression import EncryptedLinearRegression
from venumMLlib.time_series.Phineus import phineus_FFT
from venumMLlib.time_series.Phineus.phineus_FFT import rfftfreq
from venumMLlib.time_series.Phineus.phineus_rolling_average import rolling_average



def phineus_predict(ctx, data, forecast_periods=30, frequency='D', window_size=30, smoothing=False):
    data['ds'] = pd.to_datetime(data['ds'])
    data = data.set_index('ds')
    
    y = data['y'].values
    t = np.arange(len(y))
    
    # Normalization of time for the entire dataset including forecast
    t_min = t.min()
    t_max = t.max()
    forecast_extension = t_max + forecast_periods
    t_normalized = (t - t_min) / (t_max - t_min)

    t_enc = encrypt_array(t_normalized,ctx)

    if smoothing:
        y = rolling_average(ctx, y, window_size)

    # Trend modeling
    trend_model = EncryptedLinearRegression(ctx)
    trend_model.encrypted_fit(ctx, t_enc, y, lr=0.3, epochs=10)
    trend = trend_model.predict(t_enc.reshape(-1, 1), ctx)

    # # Detrend the data
    detrended_y = y - trend  
    # # Seasonal modeling
    fft_values = phineus_FFT.FFT(ctx, detrended_y, pad_data=False, window_data=False)
    sr = len(detrended_y)
    frequencies = rfftfreq(len(fft_values), 1/sr)
    magnitudes = np.asarray([(real**2 + imag**2) for real, imag in fft_values])
    magnitudes = magnitudes[:len(frequencies)]

 
    # # Prepare for predictions
    total_t = np.linspace(0, (len(y) + forecast_periods )/ len(y) , len(y) + forecast_periods)  # Generate normalized total_t for forecasting
    total_t_enc = encrypt_array(total_t,ctx)

    trend_predictions = trend_model.predict(total_t_enc.reshape(-1, 1),ctx)

   
    return fft_values, frequencies, total_t, trend_predictions




def reconstruct_signal(amplitudes, phases, frequencies, t, number_of_frequencies = 10):
    
    num_samples = len(amplitudes)/2   
    amplitudes = amplitudes[:len(frequencies)]
    indices = np.argsort(-np.abs(amplitudes))[:10]
    frequencies = frequencies[indices]
    amplitudes  = amplitudes[indices]
    phases = phases[indices]
    
    reconstructed_signal = np.zeros(len(t), dtype=np.complex64)

    for i, freq in enumerate(frequencies):
        # Correct amplitude scaling by dividing by the number of samples
        amplitude = amplitudes[i] / num_samples
        phase = phases[i] 
        component = amplitude * np.exp(1j * (2 * np.pi * freq * t + phase))
        reconstructed_signal += component

    return np.real(reconstructed_signal)



def extend_date_column(data, date_column, extension_count, frequency='D'):

    # Ensure the date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column])

    # Get the last date from the date column
    last_date = data[date_column].iloc[-1]

    # Generate additional dates starting from the day after the last date
    additional_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=extension_count, freq=frequency)

    # Combine the original dates with the new dates
    all_dates = pd.concat([data[date_column], pd.Series(additional_dates)], ignore_index=True)

    # Create a new DataFrame with the extended dates
    extended_data = pd.DataFrame({date_column: all_dates})

    return extended_data

