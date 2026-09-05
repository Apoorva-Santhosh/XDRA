# data/

This folder is intentionally empty in the repo (see `.gitignore`). Raw datasets are large and, in the case of Kaggle sources, come with their own redistribution terms — each teammate downloads their own local copy.

Expected structure once populated locally:

```
data/
├── churn/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── fraud/
│   └── creditcard.csv
└── demand/
    ├── sales_train_validation.csv
    ├── calendar.csv
    └── sell_prices.csv
```

See `GUIDELINES.md` Section 2 for exact source links and scoping rules (especially the demand store-department aggregation requirement — don't use the full M5 hierarchy).
