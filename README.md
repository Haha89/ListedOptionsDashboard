# ListedOptionDashboard
A dashboard built with Angualr, Plotly and Aggrid to visualize the Greek surface of listed European options.

This repo contains a library to price european calls and puts using Black-Scholes models. It can retrieve listed options of any underlying using [MarketWatch](https://www.marketwatch.com/).
Implied volatility is computed using market prices and greeks are then calculated.

Finally an interactive dashboard using plotly can run on your computer or on a docker container.

![image](https://github.com/Haha89/ListedOptionsDashboard/assets/45851831/2f778202-3cf7-4181-89ff-f55bc3f5f3f6)

# Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
- Docker

## Running the app locally
1. Clone the repository
```git clone https://github.com/Haha89/OptionPricing```

2. Build the docker container
```docker-compose up -d --build```

The app will be running on http://localhost:4200

# Data
The Greek surface data is pulled from a data source of your choice, you can use an API or read it from a CSV file. You will need to modify the code accordingly to retrieve the data and pass it to the Plotly Scatter3d plot.

# Built With
- Angular - Main framework used to build the dashboard
- Plotly - Used to create the Greek surface visualization
- Fastapi: Python framework to build REST api

# License
This project is licensed under the MIT License - see the LICENSE.md file for details

# Next steps
For now the pricing model is very simple. Adding jump diffusion, a dividend yield will be interesting. Adding other numerical techniques to price options like Monte-Carlo is planned.
