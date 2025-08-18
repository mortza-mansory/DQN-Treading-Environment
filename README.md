#  Deep Q-Network (DQN) Trading Visualizer

مدل **Deep Q-Network (DQN)** یک روش **یادگیری تقویتی (Reinforcement Learning)** است که از شبکه‌های عصبی عمیق برای تخمین تابع ارزش-عمل (Q-Function) استفاده می‌کند.  
این پروژه با استفاده از DQN استراتژی‌های معاملاتی **خرید، فروش و نگهداری** را در یک محیط معاملاتی بر اساس داده‌های قیمتی لایت‌کوین یاد می‌گیرد.

---

##  منطق عملکرد DQN

###  محیط و حالت‌ها (Environment & States)
- محیط معاملاتی `TradingEnvXY` شامل داده‌های قیمتی (OHLCV) از فایل **litecoin.xlsx** است.  
- حالت‌ها (States) شامل داده‌های نرمال‌سازی‌شده با **z-score** می‌باشند.

###  اقدامات (Actions)
- **نگهداری (0)** → Hold  
- **خرید (1)** → Buy  
- **فروش (2)** → Sell  

اقدام انتخاب‌شده توسط شبکه Q از لیست `DISCRETE_ACTIONS` استخراج می‌شود.

###  پاداش (Reward)
- محاسبه بر اساس بازده لگاریتمی (**log-return**)  
- اصلاح‌شده با تابع مدیریت سرمایه `money_management`  
- پاداش اضافی بر اساس:
  - صحت پیش‌بینی (±1% تغییر قیمت)  
  - سطح ریسک انتخاب‌شده (Risk Level)  
- نتیجه: سود = پاداش مثبت ✅ | زیان = پاداش منفی ❌

###  تابع Q و یادگیری
- شبکه Q (**QNetwork**) مقادیر Q را برای هر اقدام پیش‌بینی می‌کند.  
- الگوریتم **Experience Replay**:
  - ذخیره `(state, action, reward, next_state, done)` در حافظه (deque)  
  - آموزش با نمونه‌گیری تصادفی از حافظه  
- استفاده از **Target Network** برای پایداری یادگیری  
- به‌روزرسانی دوره‌ای با `update_target_model`

###  استراتژی اکتشاف-بهره‌برداری (Exploration vs Exploitation)
- الگوریتم **Epsilon-Greedy**:  
  - با احتمال `ε` → اقدام تصادفی (اکتشاف)  
  - در غیر این صورت → بهترین اقدام بر اساس Q (بهره‌برداری)  
- `ε` با نرخ `epsilon_decay` کاهش می‌یابد تا به `epsilon_min` برسد.

###  به‌روزرسانی مدل
- تابع زیان: **MSE (Mean Squared Error)**  
- بهینه‌ساز: **Adam**  
- فرمول به‌روزرسانی:  
target = reward + gamma * max(Q(next_state))
---

## 🖥 ویژگی‌های محصول (Trading Visualizer)

###  رابط کاربری (UI)
- ساخته‌شده با **Flet**  
- شامل:
- دکمه‌های **شروع، توقف، مکث**
- انتخاب مدل و پارامترها
- نمایش لاگ‌ها و معیارها در زمان واقعی  

###  نمایش معیارها (Metrics)
- تعداد خرید، فروش، نگهداری  
- سود و درصد موفقیت  
- موجودی کیف پول و دارایی‌ها  
- معاملات موفق/ناموفق  

###  نمودارها
- **Chart All** → نمایش کل قیمت واقعی  
- **Dynamic Chart** → مقایسه قیمت پیش‌بینی‌شده و واقعی (پنجره متحرک)  
- **Prediction Chart** → نمایش پیش‌بینی قیمت در کل بازه  

###  مدیریت ریسک
- انتخاب سطح ریسک: خیلی پایین ⬅➡ خیلی بالا  
- مدیریت سرمایه بر اساس `money_management`  
- در نظر گرفتن هزینه‌های تراکنش (اسپرد + کارمزد)

###  تنظیمات یادگیری
- پارامترهای قابل تنظیم:  
- `gamma`, `epsilon`, `learning_rate`, `batch_size`  
- ذخیره در **learning_settings.json**

###  ذخیره و بارگذاری مدل
- ذخیره خودکار مدل در: **dqn_model_auto_save.pt**  
- امکان بارگذاری و تست مدل‌های قبلی  
- خروجی تست شامل نمودارها + لاگ‌ها در پوشه مجزا

###  یادداشت‌ها و درس‌های آموخته‌شده
- **AI Notes** → تحلیل اقدامات و پاداش‌ها  
- **Lessons Learned** → بازخورد معاملات موفق/ناموفق  

---

##  فرآیند یادگیری DQN

1. **شروع آموزش**  
 - داده‌های نرمال‌سازی‌شده از litecoin.xlsx  
 - شروع از تاریخ مشخص (مثلاً `2024-01-01` تا `2025-01-01`)

2. **انتخاب اقدام**  
 - تصادفی (اکتشاف) یا بهترین Q (بهره‌برداری)  
 - اقدامات: خرید، فروش، نگهداری

3. **محاسبه پاداش**  
 - بر اساس log-return  
 - صحیح بودن پیش‌بینی (±1%)  
 - اصلاح توسط `money_management`

4. **ذخیره تجربه**  
 - `(state, action, reward, next_state, done)` → حافظه عامل

5. **بازپخش تجربه (Replay)**  
 - نمونه‌گیری تصادفی از حافظه  
 - آموزش با MSE و هدف Q  
 - به‌روزرسانی دوره‌ای شبکه هدف

6. **به‌روزرسانی Epsilon**  
 - کاهش تدریجی `ε` → حرکت به سمت بهره‌برداری

7. **نظارت و بازخورد**  
 - نمایش سود، درصد موفقیت، معاملات  
 - ثبت AI Notes و Lessons Learned  

8. **توقف و ذخیره**  
 - توقف دستی یا خودکار (اتمام داده‌ها / ورشکستگی)  
 - ذخیره مدل نهایی برای تست‌های بعدی  
---
کد نویسی توسط: مرتضی منصوری
#  Deep Q-Network (DQN) Trading Visualizer

The **Deep Q-Network (DQN)** is a **Reinforcement Learning** method that uses deep neural networks to approximate the Q-function (action-value function).  
In this project, DQN is used to learn trading strategies **(Buy, Sell, Hold)** in a trading environment based on Litecoin price data.

---

##  DQN Logic

###  Environment & States
- The trading environment `TradingEnvXY` loads OHLCV data from **litecoin.xlsx**.  
- States are normalized features (z-score) of this price data.

###  Actions
- **Hold (0)**  
- **Buy (1)**  
- **Sell (2)**  

Actions are selected by the Q-Network from the list `DISCRETE_ACTIONS`.

###  Reward
- Calculated from **log-return**  
- Adjusted by the money management function `money_management`  
- Extra reward depends on:
  - Prediction accuracy (±1% price change)  
  - User-selected **Risk Level**  
- Outcome: Profit = Positive reward ✅ | Loss = Negative reward ❌

###  Q-Function & Learning
- The Q-Network (**QNetwork**) estimates Q-values for each action.  
- **Experience Replay** algorithm:  
  - Stores `(state, action, reward, next_state, done)` in memory (deque)  
  - Samples random batches for training  
- A **Target Network** is used for stability.  
- Updated periodically with `update_target_model`.

###  Exploration vs Exploitation
- **Epsilon-Greedy strategy**:  
  - With probability `ε` → choose a random action (exploration)  
  - Otherwise → choose the best Q-value action (exploitation)  
- `ε` decays over time with `epsilon_decay` until `epsilon_min` is reached.

###  Model Update
- Loss function: **MSE (Mean Squared Error)**  
- Optimizer: **Adam**  
- Update rule:
target = reward + gamma * max(Q(next_state))

---

##  Trading Visualizer Features

###  User Interface (UI)
- Built with **Flet**  
- Includes:  
- **Start, Stop, Pause** buttons  
- Model and parameter selection  
- Real-time logs and metrics  

###  Metrics
- Number of Buys, Sells, Holds  
- Profit and success rate  
- Wallet balance and assets  
- Successful / Failed trades  

###  Charts
- **Chart All** → Real price over the full timeline  
- **Dynamic Chart** → Predicted vs. real price in a moving window  
- **Prediction Chart** → Price prediction across the whole dataset  

###  Risk Management
- User can select risk level: Very Low ⬅➡ Very High  
- Money management ensures correct position sizing  
- Includes transaction costs (spread + fees)

###  Learning Settings
- Adjustable parameters:  
- `gamma`, `epsilon`, `learning_rate`, `batch_size`  
- Saved in **learning_settings.json**

###  Save & Load Model
- Automatic save: **dqn_model_auto_save.pt**  
- Load and test saved models  
- Test results (charts + logs) saved in separate folders

###  AI Notes & Lessons Learned
- **AI Notes** → track actions and rewards  
- **Lessons Learned** → analyze successful and failed trades  

---

##  DQN Training Process

1. **Training Start**  
 - Normalized data from **litecoin.xlsx**  
 - Example range: `2024-01-01` to `2025-01-01`

2. **Action Selection**  
 - Random (exploration) or best Q (exploitation)  
 - Actions: Buy, Sell, Hold

3. **Reward Calculation**  
 - Based on log-return  
 - Bonus if prediction (±1%) is correct  
 - Adjusted with `money_management`

4. **Experience Storage**  
 - Save `(state, action, reward, next_state, done)` in agent memory

5. **Replay Training**  
 - Sample random experiences  
 - Train with MSE and Q targets  
 - Periodically update target network  

6. **Epsilon Update**  
 - Gradual decay of `ε` → more exploitation over time  

7. **Monitoring & Feedback**  
 - Show profit, success rate, trades  
 - Log AI Notes and Lessons Learned  

8. **Stopping & Saving**  
 - Stop manually or automatically (no data / no balance)  
 - Save final model for later testing  

---

Developed by: **Morteza Mansouri**

