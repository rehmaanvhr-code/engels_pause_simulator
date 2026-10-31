# Engels' Pause Simulator

A small macro simulation of the mechanism behind Engels' pause. A technology shock raises output per worker, but wages stay flat for decades because the new capital is scarce and owned by a few, so the surplus flows to owners. Wages only catch up once capital ownership spreads, or once politics intervenes through redistribution or labour bargaining. The model is calibrated against British real wages and output per worker from 1760 to 1870, then rerun with the diffusion period compressed from decades into years to see what the same mechanism looks like at AI speed.

## Connection to the research

This is the code companion to "The Age of AI and the Industrial Revolution: A Comparative Study of Technological Transformation and Inequality" (Rehman, 2026). The paper argues that AI repeats the Industrial Revolution's distributional mechanism: owners of the scarce factor capture the surplus until scarcity eases or policy steps in. The paper makes the argument with history and the figure of British wages against output. This repo makes it a model with parameters, so the claim can be stress tested instead of asserted. The follow up papers on the governance lag and on who bears the cost of waiting build on the same mechanism.

## Why it matters for AI governance in financial services

Impact assessment is becoming a standard part of responsible AI frameworks, and the workforce and distributional effects of automation are part of that. A simulation like this gives a governance team a way to reason about scenarios (how fast a capability diffuses, who owns it, what policy levers exist) rather than relying on narrative alone. The same structure can be reused for internal questions such as how quickly productivity gains from an AI deployment reach different groups of employees or customers.

## How it works

Each year the economy produces output from technology, capital and labour. Technology follows a logistic path from 1 to 1 plus the shock size over the diffusion period. The competitive wage is the marginal product of labour, but the actual wage moves toward it only at a speed that depends on how widely capital is owned and on any bargaining power workers hold. Profits, net of any redistribution, are reinvested as capital. A thousand agents are tracked so an income Gini can be computed each year, with owners holding capital according to a Pareto draw.

`calibrate` runs a small grid search over shock size, diffusion period and the rate at which ownership spreads, minimizing squared error against the decade level historical series. `pause_length` measures how many years the wage share stays depressed below its starting level. `owner_surplus_share` is the share of the cumulative output gain that went to profits rather than wages.

## Run

```
pip install -r requirements.txt
python simulate.py
pytest
```

Outputs land in `outputs/`:

`engels_pause_baseline.png` shows simulated and historical wage and output indices for 1760 to 1870.

`ai_scenarios.png` shows the wage index and Gini under five scenarios over 60 years: the historical pace, AI speed with no policy response, AI speed with broad capital ownership, AI speed with 20 percent redistribution of profits, and AI speed with stronger bargaining.

`summary.csv` reports the pause length, the owner share of the surplus, the wage index at year 20 and year 59, and the peak Gini for each scenario.

The headline from the default run is that compressing diffusion on its own does not shorten the pause, it raises inequality faster and leaves the wage share depressed for longer than the horizon. Spreading ownership or giving workers bargaining power cuts the pause to under 20 years. Redistribution lifts wages early but leaves the share of surplus going to owners high, because it treats the symptom and not the scarcity.

## Data

`data/british_wages_output_1760_1870.csv` holds decade level index values for real wages and output per worker, 1760 = 100. These are approximate transcriptions in the spirit of the series in Robert Allen's work on Engels' pause (Allen, 2009, Explorations in Economic History) and should be replaced with the published series for any serious use. They are here so the repo runs without downloads.

## Limitations

The model is deliberately simple. There is one sector, no trade, no population growth and no technological change beyond the single shock. Labour supply is fixed. The calibration is a coarse grid search on a handful of points. The AI scenarios reuse the historical shock size, which is a strong assumption. The point of the repo is the mechanism, not a forecast.

## Author

Saif Rehman, MMAI, Smith School of Business, Queen's University.
