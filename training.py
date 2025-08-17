import asyncio
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from Q import QNetwork, DISCRETE_ACTIONS, NUM_ACTIONS, DQNAgent
from tradingenv.env import TradingEnvXY
from strategy import money_management, calculate_success_percentage, calculate_financial_success
import flet as ft
from flet import Colors
import json

ACTION_NAMES = {0: "Hold", 1: "Buy", 2: "Sell"}

def highlight_metric(field, color, duration=1.0):
    field.border_color = color
    field.border_width = 2
    field.update()

    async def clear():
        await asyncio.sleep(duration)
        field.border_color = Colors.TRANSPARENT
        field.border_width = 1
        field.update()

    asyncio.create_task(clear())

def predict_future_price(df, step, forecast_steps):
    """Predict price for the next 'forecast_steps' steps using simple averaging."""
    if step >= len(df) - 1:
        return df["priceClose"].iloc[-1]
    end_idx = min(step + forecast_steps, len(df) - 1)
    future_prices = df["priceClose"].iloc[step:end_idx]
    if len(future_prices) == 0:
        return df["priceClose"].iloc[step]
    return future_prices.mean()

async def run_training(
    page, status, log_text, ai_notes_text, lessons_text, metrics,
    chart_all, chart_dynamic, chart_pred, ax_all, ax_dynamic, ax_pred,
    count, delay, training_manager, df, X, Y, initial_cash, risk_level,
    set_agent=None
):
    from strategy import money_management, calculate_success_percentage, calculate_financial_success
    from Q import DQNAgent, NUM_ACTIONS
    from tradingenv.env import TradingEnvXY
    import numpy as np

    ACTION_NAMES = {0: "Hold", 1: "Buy", 2: "Sell"}

    try:
        initial_cash = float(initial_cash)
    except ValueError:
        initial_cash = 1000
        log_text.value += "Invalid portfolio amount. Using default (1000).\n"
        log_text.update()

    # Load learning settings from JSON
    settings_file = "learning_settings.json"
    learning_settings = {
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "learning_rate": 0.001,
        "batch_size": 32,
        "forecast_steps": 1
    }
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r") as f:
                learning_settings.update(json.load(f))
            log_text.value += "Loaded learning settings from learning_settings.json\n"
        except Exception as e:
            log_text.value += f"Error loading learning settings: {e}\n"
        log_text.update()

    agent = DQNAgent(
        input_shape=(X.shape[1],),
        action_space=NUM_ACTIONS,
        gamma=learning_settings["gamma"],
        epsilon=learning_settings["epsilon"],
        epsilon_min=learning_settings["epsilon_min"],
        epsilon_decay=learning_settings["epsilon_decay"],
        learning_rate=learning_settings["learning_rate"]
    )
    if set_agent:
        set_agent(agent)

    env = TradingEnvXY(X=X, Y=Y, transformer="z-score", reward="logret",
                       cash=initial_cash, spread=0.0001, markup=0.002,
                       fee=0.0001, fixed=0.01)

    # Initialize
    portfolio = initial_cash
    current_assets = 0
    x_prog, y_prog, actual_prices = [], [], []
    step, episode = 0, 0
    buys = sells = holds = 0
    window_size = 50
    prev_portfolio = initial_cash
    successful_trades = failed_trades = 0
    financial_successes = []

    # Initial chart drawing
    ax_all.clear()
    ax_all.plot(df.index, df["priceClose"], color="gray", label="Actual Price")
    ax_all.set_title("Actual Price")
    ax_all.legend()
    chart_all.update()

    ax_dynamic.clear()
    ax_dynamic.set_title("Dynamic Predicted vs Actual Price")
    ax_dynamic.set_xlabel("Steps")
    ax_dynamic.set_ylabel("Price")
    ax_dynamic.grid(True)
    chart_dynamic.update()

    ax_pred.clear()
    ax_pred.plot(df.index, df["priceClose"], color="gray", alpha=0.5)
    ax_pred.set_title("Predicted Price (Full Timeline)")
    chart_pred.update()

    # Status update
    status.value = "Training Started"
    status.color = Colors.GREEN
    status.update()

    while training_manager["training_active"]:
        state = env.reset()
        state = np.reshape(state, (1, -1))
        done = False

        for _ in range(count):
            if not training_manager["training_active"]:
                log_text.value += f"Training stopped at step {step}.\n"
                log_text.update()
                break

            if training_manager["pause_requested"]:
                while training_manager["pause_requested"] and training_manager["training_active"]:
                    await asyncio.sleep(0.1)
                    page.update()

            if step >= len(df):
                done = True
                log_text.value += f"Episode {episode}: Reached end of data at step {step}.\n"
                log_text.update()
                break

            action_idx, act = agent.act(state)
            next_state, reward, done, info = env.step(act)
            reward /= 100
            current_price = df["priceClose"].iloc[min(step, len(df)-1)]

            # Price prediction using multiple steps
            predicted_price = predict_future_price(df, step, learning_settings["forecast_steps"])
            if action_idx == 1:  # Buy
                predicted_price *= 1.01
            elif action_idx == 2:  # Sell
                predicted_price *= 0.99

            # Money management
            mm_reward, mm_done, trade_amount, units_traded = money_management(
                initial_cash, action_idx, risk_level, current_price, portfolio, current_assets, log_text,
                predicted_price=predicted_price
            )
            reward += mm_reward
            if mm_done:
                done = True
                lesson = f"Episode {episode}: Training stopped - portfolio depleted."
                lessons_text.value += f"{lesson}\n"
                lessons_text.update()

            # Apply trade
            if action_idx == 1 and trade_amount > 0:
                portfolio -= trade_amount
                current_assets += units_traded
            elif action_idx == 2 and trade_amount > 0:
                portfolio += trade_amount
                current_assets -= units_traded

            # Save state
            agent.remember(state, action_idx, reward, next_state, done)
            state = np.reshape(next_state, (1, -1))

            # Metrics logic
            profit_delta = portfolio - prev_portfolio
            if trade_amount > 0:
                if profit_delta > 0:
                    successful_trades += 1
                else:
                    failed_trades += 1

            x_prog.append(step)
            y_prog.append(predicted_price)
            actual_prices.append(current_price)

            if action_idx == 0:
                holds += 1
                highlight_metric(metrics["Hold"], Colors.GREY)
            elif action_idx == 1:
                buys += 1
                highlight_metric(metrics["Buys"], Colors.GREEN)
            elif action_idx == 2:
                sells += 1
                highlight_metric(metrics["Sells"], Colors.RED)

            # Update metrics
            metrics_values = {
                "Buys": str(buys),
                "Sells": str(sells),
                "Hold": str(holds),
                "Profit": f"{portfolio - initial_cash:.2f}",
                "% Success": f"{calculate_success_percentage(portfolio, initial_cash, df):.1f}%",
                "% Financial Success": f"{calculate_financial_success(portfolio, initial_cash, df):.1f}%",
                "Current Money in Wallet": f"{portfolio:.2f}",
                "Assets": f"{current_assets:.4f}",
                "Steps": str(step),
                "Epsilon": f"{agent.epsilon:.3f}",
                "Successful Trades": str(successful_trades),
                "Failed Trades": str(failed_trades),
            }

            for key, value in metrics_values.items():
                metrics[key].value = value
                metrics[key].update()

            # Update charts every 10 steps
            if step % 10 == 0:
                # Chart Dynamic
                start = max(0, len(x_prog) - window_size)
                ax_dynamic.clear()
                ax_dynamic.plot(x_prog[start:], y_prog[start:], label="Predicted")
                ax_dynamic.plot(x_prog[start:], actual_prices[start:], label="Actual", linestyle="--")
                ax_dynamic.legend()
                ax_dynamic.grid(True)
                chart_dynamic.update()

                # Chart Pred
                ax_pred.clear()
                ax_pred.plot(df.index, df["priceClose"], color="gray", alpha=0.5)
                plot_length = min(len(y_prog), len(df.index))
                ax_pred.plot(df.index[:plot_length], y_prog[:plot_length], label="Predicted Price")
                ax_pred.legend()
                chart_pred.update()

                # Chart All with span
                ax_all.clear()
                ax_all.plot(df.index, df["priceClose"], color="gray")
                span_start = df.index[max(0, step - window_size)]
                span_end = df.index[min(step, len(df)-1)]
                ax_all.axvspan(span_start, span_end, color="red", alpha=0.3)
                ax_all.set_title("Actual Price with Training Span")
                chart_all.update()

            await asyncio.sleep(delay)
            page.update()

            if done:
                break

            prev_portfolio = portfolio
            step += 1

        if training_manager["training_active"]:
            agent.replay(batch_size=learning_settings["batch_size"])
            episode += 1
            log_text.value += f"Episode {episode} completed.\n"
            log_text.update()

    if training_manager["training_active"]:
        agent.save_model("dqn_model_final.pt")
        log_text.value += "Model saved as 'dqn_model_final.pt'.\n"
        log_text.update()

    # Define return functions
    def update_mode(mode): log_text.value += f"Mode updated to {mode}\n"; log_text.update()
    def update_initial_cash(cash): log_text.value += f"Initial portfolio updated to {cash}\n"; log_text.update()
    def update_risk_level(risk): log_text.value += f"Risk level updated to {risk}\n"; log_text.update()
    def clear_ai_notes(): ai_notes_text.value = ""; ai_notes_text.update()
    def clear_lessons(): lessons_text.value = ""; lessons_text.update()

    return update_mode, update_initial_cash, update_risk_level, clear_ai_notes, clear_lessons