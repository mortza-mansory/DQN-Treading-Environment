import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import flet as ft

def load_data(start_date="2024-01-01", end_date="2025-01-01"):
    df = pd.read_excel("litecoin.xlsx", engine="openpyxl")
    df["timeClose"] = pd.to_datetime(df["timeClose"], unit="ms")
    df.set_index("timeClose", inplace=True)
    df.sort_index(inplace=True)
    if start_date not in df.index or end_date not in df.index:
        start_date = df.index[df.index >= start_date][0]
        end_date = df.index[df.index <= end_date][-1]
    df = df.loc[start_date:end_date]
    Y = df[["priceClose"]].rename(columns={"priceClose": "LTC"})
    X_raw = df[["priceOpen", "priceHigh", "priceLow", "priceClose", "volume"]].values
    mean, std = X_raw.mean(0), X_raw.std(0) + 1e-5
    X = (X_raw - mean) / std
    X = pd.DataFrame(X, index=df.index,
                     columns=["priceOpen", "priceHigh", "priceLow", "priceClose", "volume"])
    return df, X, Y

def calculate_ideal_profit(df, initial_cash):
    min_p, max_p = df["priceClose"].min(), df["priceClose"].max()
    ideal = (max_p - min_p) * (initial_cash / min_p)
    return max(ideal, 1e-5)

def calculate_financial_success(portfolio, initial_cash, df):
    portfolio_return = (portfolio - initial_cash) / initial_cash * 100
    if len(df) > 1:
        market_start_price = df["priceClose"].iloc[0]
        market_end_price = df["priceClose"].iloc[-1]
        market_return = (market_end_price - market_start_price) / market_start_price * 100
    else:
        market_return = 0
    if market_return != 0:
        financial_success = (portfolio_return / market_return) * 100
    else:
        financial_success = portfolio_return
    return financial_success

def calculate_success_percentage(portfolio, initial_cash, df):
    actual_profit = portfolio - initial_cash
    ideal_profit = calculate_ideal_profit(df, initial_cash)
    if ideal_profit == 0:
        return 0.0
    success = (actual_profit / ideal_profit) * 100
    return min(max(success, 0), 100)

def save_period_results(period_start, period_end, x_prog, y_prog, actual_prices, log_data, ax_zoom, plot_number, df):
    folder_name = f"plot{plot_number}_{period_start[:4]}_{period_end[:4]}"
    os.makedirs(folder_name, exist_ok=True)
    
    plt.figure(figsize=(16, 9))
    plt.plot(df.index[:len(actual_prices)], actual_prices, label="Actual Price", color="green")
    plt.plot(df.index[:len(y_prog)], y_prog, label="Predicted Price", color="blue")
    plt.title(f"Actual vs Predicted Prices ({period_start[:4]}-{period_end[:4]})")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_name, "price_comparison.png"), dpi=400)
    plt.close()
    
    with open(os.path.join(folder_name, "debug_log.json"), "w") as f:
        json.dump(log_data, f, indent=4)

def money_management(initial_cash, action_idx, risk_level, current_price, portfolio, current_assets, log_text=None, predicted_price=None):
    reward_modifier = 0
    force_done = False
    trade_amount = 0
    units_traded = 0

    try:
        initial_cash = float(initial_cash)
        portfolio = float(portfolio)
        current_assets = float(current_assets)
        current_price = float(current_price)
        predicted_price = float(predicted_price) if predicted_price is not None else current_price
    except (TypeError, ValueError) as e:
        if log_text:
            log_text.value += f"Invalid input in money_management: {e}\n"
            log_text.update()
        return reward_modifier, force_done, trade_amount, units_traded

    risk_percentages = {
        "Very High Risk": 0.80,
        "High Risk": 0.65,
        "Medium Risk": 0.50,
        "Low Risk": 0.35,
        "Very Low Risk": 0.20
    }
    max_trade_percentage = risk_percentages.get(risk_level, 0.50)

    if portfolio <= 0:
        reward_modifier = -100
        force_done = True
        if log_text:
            log_text.value += "Portfolio depleted: Force done.\n"
            log_text.update()
        return reward_modifier, force_done, trade_amount, units_traded

    if action_idx == 1:  # Buy
        max_trade_amount = portfolio * max_trade_percentage
        transaction_cost = 0.01 + (current_price * (0.0001 + 0.002 + 0.0001))

        if max_trade_amount < transaction_cost:
            reward_modifier = -10
            if log_text:
                log_text.value += f"Buy skipped: Not enough for transaction cost ({transaction_cost:.2f})\n"
                log_text.update()
        else:
            units_traded = max((max_trade_amount - transaction_cost) / current_price, 0)
            trade_amount = units_traded * current_price + transaction_cost

            if units_traded <= 0 or trade_amount > portfolio:
                reward_modifier = -10
                if log_text:
                    log_text.value += f"Buy skipped: Insufficient portfolio value for trade.\n"
                    log_text.update()
                units_traded = 0
                trade_amount = 0
            else:
                # Adjust reward based on predicted price
                expected_profit = (predicted_price - current_price) * units_traded
                reward_modifier = 1.0 + expected_profit / trade_amount if trade_amount > 0 else 1.0
                if log_text:
                    log_text.value += f"Buy: {units_traded:.4f} units at {current_price:.2f}, cost={trade_amount:.2f}, expected profit={expected_profit:.2f}\n"
                    log_text.update()

    elif action_idx == 2:  # Sell
        if current_assets > 0:
            units_traded = min(current_assets * max_trade_percentage, current_assets)
            transaction_cost = 0.01 + (current_price * units_traded * (0.0001 + 0.002 + 0.0001))
            trade_amount = units_traded * current_price - transaction_cost

            if trade_amount <= 0:
                reward_modifier = -10
                if log_text:
                    log_text.value += f"Sell skipped: Invalid trade amount ({trade_amount:.2f})\n"
                    log_text.update()
                units_traded = 0
                trade_amount = 0
            else:
                # Adjust reward based on predicted price
                expected_profit = (current_price - predicted_price) * units_traded
                reward_modifier = 1.0 + expected_profit / trade_amount if trade_amount > 0 else 1.0
                if log_text:
                    log_text.value += f"Sell: {units_traded:.4f} units at {current_price:.2f}, revenue={trade_amount:.2f}, expected profit={expected_profit:.2f}\n"
                    log_text.update()
        else:
            reward_modifier = -5
            if log_text:
                log_text.value += "Sell skipped: No assets.\n"
                log_text.update()

    elif action_idx == 0:  # Hold
        if log_text:
            log_text.value += "Hold: No trade.\n"
            log_text.update()

    return reward_modifier, force_done, trade_amount, units_traded