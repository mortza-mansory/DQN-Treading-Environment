import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from Q import QNetwork, DISCRETE_ACTIONS, NUM_ACTIONS
from tradingenv.env import TradingEnvXY
from strategy import money_management, calculate_success_percentage, calculate_financial_success
import flet as ft

ACTION_NAMES = {0: "Hold", 1: "Buy", 2: "Sell"}

def load_and_test_model(model_path, log_text, metrics, chart_all, chart_dynamic, chart_pred, ax_all, ax_dynamic, ax_pred, lessons_text, ai_notes_text, df, X, Y, initial_cash, risk_level):
    try:
        initial_cash = float(initial_cash)
    except ValueError:
        initial_cash = 1000
        log_text.value += "Invalid portfolio amount. Using default (1000).\n"
        log_text.update()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QNetwork(input_shape=(X.shape[1],), action_space=NUM_ACTIONS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    env = TradingEnvXY(X=X, Y=Y, transformer="z-score", reward="logret", cash=initial_cash,
                       spread=0.0001, markup=0.002, fee=0.0001, fixed=0.01)

    portfolio = initial_cash
    current_assets = 0
    buys = sells = holds = 0
    successful_trades = failed_trades = 0
    x_prog = []
    y_prog = []
    actual_prices = []
    prev_portfolio = initial_cash
    step = 0
    state = env.reset()
    state = np.reshape(state, (1, *X.shape[1:]))
    done = False
    seen_notes = set()
    seen_lessons = set()
    hold_streak = 0
    financial_successes = []

    # Initialize charts
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
    ax_pred.set_title("Predicted Price (Full Timeline)")
    chart_pred.update()

    while not done:
        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = model(state_tensor).cpu().numpy()[0]
        idx = np.argmax(q_values)
        if idx == 2 and current_assets <= 0:
            idx = 0  # Fallback to Hold
        act = DISCRETE_ACTIONS[idx]

        try:
            nxt, r, done, info = env.step(act)
        except Exception as e:
            log_text.value += f"Error during test: {e}\n"
            log_text.update()
            break

        r = r / 100
        current_price = df["priceClose"].iloc[step] if step < len(df) else df["priceClose"].iloc[-1]
        predicted_price = current_price
        if idx == 1:
            predicted_price *= 1.01
            if predicted_price > current_price:
                r += 0.1
            else:
                lesson = f"Step {step}: Buy action predicted price increase ({predicted_price:.2f}) but actual price was {current_price:.2f}. Reason: Incorrect price prediction."
                if lesson not in seen_lessons:
                    seen_lessons.add(lesson)
                    lessons_text.value += f"{lesson}\n"
                    lines = lessons_text.value.split("\n")
                    if len(lines) > 100:
                        lessons_text.value = "\n".join(lines[-100:])
                    lessons_text.update()
        elif idx == 2:
            predicted_price *= 0.99
            if predicted_price < current_price:
                r += 0.1
            else:
                lesson = f"Step {step}: Sell action predicted price decrease ({predicted_price:.2f}) but actual price was {current_price:.2f}. Reason: Incorrect price prediction."
                if lesson not in seen_lessons:
                    seen_lessons.add(lesson)
                    lessons_text.value += f"{lesson}\n"
                    lines = lessons_text.value.split("\n")
                    if len(lines) > 100:
                        lessons_text.value = "\n".join(lines[-100:])
                    lessons_text.update()

        mm_reward, mm_done, trade_amount, units_traded = money_management(
            initial_cash, idx, risk_level, current_price, portfolio, current_assets, log_text
        )
        r += mm_reward
        if mm_done:
            lesson = f"Test Failed: Portfolio depleted at step {step}. Reason: Portfolio reached zero due to excessive losses."
            if lesson not in seen_lessons:
                seen_lessons.add(lesson)
                lessons_text.value += f"{lesson}\n"
                lines = lessons_text.value.split("\n")
                if len(lines) > 100:
                    lessons_text.value = "\n".join(lines[-100:])
                lessons_text.update()
                log_text.value += f"{lesson}\n"
                log_text.update()
            break

        if idx == 1 and trade_amount > 0:
            portfolio -= trade_amount
            current_assets += units_traded
        elif idx == 2 and trade_amount > 0:
            portfolio += trade_amount
            current_assets -= units_traded

        portfolio = info.get('net_liquidation_value', portfolio)

        if trade_amount > 0:
            profit_delta = portfolio - prev_portfolio
            if profit_delta > 0:
                successful_trades += 1
                lesson = f"Step {step}: Successful trade ({ACTION_NAMES[idx]}), Profit={profit_delta:.2f}. Reason: Correct price movement prediction."
                if lesson not in seen_lessons:
                    seen_lessons.add(lesson)
                    lessons_text.value += f"{lesson}\n"
                    lines = lessons_text.value.split("\n")
                    if len(lines) > 100:
                        lessons_text.value = "\n".join(lines[-100:])
                    lessons_text.update()
            else:
                failed_trades += 1
                lesson = f"Step {step}: Failed trade ({ACTION_NAMES[idx]}), Loss={profit_delta:.2f}. Reason: Incorrect price movement or high transaction costs."
                if lesson not in seen_lessons:
                    seen_lessons.add(lesson)
                    lessons_text.value += f"{lesson}\n"
                    lines = lessons_text.value.split("\n")
                    if len(lines) > 100:
                        lessons_text.value = "\n".join(lines[-100:])
                    lessons_text.update()

        if idx == 0:
            hold_streak += 1
            if hold_streak >= 20 and abs(portfolio - initial_cash) > initial_cash * 0.1:
                lesson = f"Step {step}: Long hold streak (20+ steps). Suggestion: Consider trading to capitalize on market movements."
                if lesson not in seen_lessons:
                    seen_lessons.add(lesson)
                    lessons_text.value += f"{lesson}\n"
                    lines = lessons_text.value.split("\n")
                    if len(lines) > 100:
                        lessons_text.value = "\n".join(lines[-100:])
                    lessons_text.update()
                hold_streak = 0
        else:
            hold_streak = 0

        if r < -10:
            lesson = f"Step {step}: Large negative reward ({r:.3f}) for {ACTION_NAMES[idx]}. Suggestion: Adjust strategy to avoid high-risk trades."
            if lesson not in seen_lessons:
                seen_lessons.add(lesson)
                lessons_text.value += f"{lesson}\n"
                lines = lessons_text.value.split("\n")
                if len(lines) > 100:
                    lessons_text.value = "\n".join(lines[-100:])
                lessons_text.update()

        if abs(r) > 0.5 or trade_amount > 0:
            max_q = np.max(q_values)
            note = f"Step {step}: Action={ACTION_NAMES[idx]}, Reward={r:.3f}, Profit={(portfolio - initial_cash):.2f}, Max Q={max_q:.3f}"
            if note not in seen_notes:
                seen_notes.add(note)
                ai_notes_text.value += f"{note}\n"
                lines = ai_notes_text.value.split("\n")
                if len(lines) > 500:
                    ai_notes_text.value = "\n".join(lines[-500:])
                ai_notes_text.update()

        x_prog.append(step)
        y_prog.append(predicted_price)
        actual_prices.append(current_price)

        success_pct = calculate_success_percentage(portfolio, initial_cash, df)
        profit = portfolio - initial_cash
        financial_success = calculate_financial_success(portfolio, initial_cash, df)
        financial_successes.append(financial_success)
        avg_financial_success = sum(financial_successes) / len(financial_successes) if financial_successes else 0

        metrics_values = {
            "Buys": str(buys),
            "Sells": str(sells),
            "Hold": str(holds),
            "Profit": f"{profit:.2f}",
            "% Success": f"{success_pct:.1f}%",
            "Current Money in Wallet": f"{portfolio:.2f}",
            "Assets": f"{current_assets:.4f}",
            "% Financial Success": f"{financial_success:.1f}%",
            "Average Financial Success": f"{avg_financial_success:.1f}%",
            "Steps": str(step),
            "Epsilon": "0.000",
            "Successful Trades": str(successful_trades),
            "Failed Trades": str(failed_trades)
        }
        
        log_message = (f"Step {step}: Portfolio={portfolio:.2f}, Assets={current_assets:.4f}, "
                       f"Reward={r:.4f}, Action={ACTION_NAMES[idx]}, TradeAmount={trade_amount:.2f}, "
                       f"UnitsTraded={units_traded:.4f}, Predicted Price={predicted_price:.2f}, "
                       f"Actual Price={current_price:.2f}")
        log_text.value += f"{log_message}\n"
        lines = log_text.value.split("\n")
        if len(lines) > 1000:
            log_text.value = "\n".join(lines[-1000:])
        log_text.update()

        
        
        if step % 10 == 0:
            ax_dynamic.clear()
            ax_dynamic.plot(x_prog, y_prog, label="Predicted Price", color="blue")
            ax_dynamic.plot(x_prog, actual_prices, label="Actual Price", color="green")
            ax_dynamic.grid(True)
            ax_dynamic.set_title("Dynamic Predicted vs Actual Price")
            ax_dynamic.set_xlabel("Steps")
            ax_dynamic.set_ylabel("Price")
            ax_dynamic.legend()
            chart_dynamic.update()

            ax_pred.clear()
            if y_prog:
                ax_pred.plot(df.index[:len(y_prog)], y_prog, label="Predicted Price", color="blue")
                ax_pred.legend()
            ax_pred.set_title("Predicted Price (Full Timeline)")
            chart_pred.update()

            start_idx = max(0, step - 20)
            end_idx = min(step, len(df.index) - 1)
            ax_all.clear()
            ax_all.plot(df.index, df["priceClose"], color="gray", label="Actual Price")
            ax_all.axvspan(df.index[start_idx], df.index[end_idx], color="red", alpha=0.3)
            ax_all.legend()
            chart_all.update()

        for key, value in metrics_values.items():
            metrics[key].value = value
            metrics[key].update()

        state = np.reshape(nxt, (1, *X.shape[1:]))
        prev_portfolio = portfolio
        step += 1
        if idx == 0:
            holds += 1
        elif idx == 1 and trade_amount > 0:
            buys += 1
        elif idx == 2 and trade_amount > 0:
            sells += 1

    log_text.value += f"Test Result: Profit={profit:.2f}, Success={success_pct:.1f}%, Buys={buys}, Sells={sells}, Holds={holds}\n"
    log_text.update()

    folder_name = f"test_model_{int(time.time())}"
    os.makedirs(folder_name, exist_ok=True)
    plt.figure(figsize=(16, 9))
    plt.plot(x_prog, actual_prices, label="Actual Price", color="green")
    plt.plot(x_prog, y_prog, label="Predicted Price", color="blue")
    plt.title("Test: Actual vs Predicted Prices")
    plt.xlabel("Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_name, "price_comparison.png"), dpi=400)
    plt.close()

    with open(os.path.join(folder_name, "test_log.txt"), "w") as f:
        f.write(log_text.value)

    with open(os.path.join(folder_name, "lessons_learned.txt"), "w") as f:
        f.write(lessons_text.value)