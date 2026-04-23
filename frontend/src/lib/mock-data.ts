import { AegisResult } from "./types";

export const MOCK_RESULT: AegisResult = {
  dataset_analysis: {
    bias_report: {
      distribution_bias: {
        sex: {
          group_proportions: { Female: 0.3315, Male: 0.6685 },
          imbalance_ratio: 2.016,
        },
        race: {
          group_proportions: {
            White: 0.855,
            Black: 0.0959,
            "Asian-Pac": 0.0311,
            "Amer-Indian": 0.0096,
            Other: 0.0083,
          },
          imbalance_ratio: 102.86,
        },
        marital_status: {
          group_proportions: {
            "Married-civ": 0.4582,
            "Never-married": 0.33,
            Divorced: 0.1358,
            Separated: 0.0313,
            Widowed: 0.0311,
            "Married-spouse-absent": 0.0129,
            "Married-AF": 0.0008,
          },
          imbalance_ratio: 604.84,
        },
      },
      outcome_bias: {
        sex: {
          outcome_rates: { Female: 0.1093, Male: 0.3038 },
          disparity: 0.1945,
        },
        race: {
          outcome_rates: {
            White: 0.254,
            Black: 0.1208,
            "Asian-Pac": 0.2693,
            "Amer-Indian": 0.117,
            Other: 0.1232,
          },
          disparity: 0.1522,
        },
        marital_status: {
          outcome_rates: {
            "Married-civ": 0.4461,
            "Never-married": 0.0455,
            Divorced: 0.1012,
            Separated: 0.0647,
            Widowed: 0.0843,
            "Married-spouse-absent": 0.0924,
            "Married-AF": 0.3784,
          },
          disparity: 0.4007,
        },
      },
      fairness_metrics: {
        sex: {
          demographic_parity_difference: 0.1945,
          disparate_impact_ratio: 0.3597,
        },
        race: {
          demographic_parity_difference: 0.1522,
          disparate_impact_ratio: 0.4346,
        },
        marital_status: {
          demographic_parity_difference: 0.4007,
          disparate_impact_ratio: 0.1019,
        },
      },
      advanced_bias: {
        proxy_bias: {
          relationship: { correlation: 0.5798, correlated_with: "sex" },
        },
        intersectional_bias: {
          features: ["marital_status", "race"],
          max_disparity: 1.0,
          group_rates: {
            "Married-civ + White": 0.4539,
            "Married-civ + Black": 0.3563,
            "Never-married + White": 0.0489,
            "Never-married + Black": 0.0217,
            "Divorced + White": 0.1049,
            "Divorced + Black": 0.0762,
          },
        },
        label_bias: {},
      },
      insights: [
        "Representation imbalance in 'sex': ratio = 2.02",
        "Outcome disparity in 'sex': gap = 19.45%",
        "Representation imbalance in 'race': ratio = 102.86",
        "Outcome disparity in 'race': gap = 15.22%",
        "Representation imbalance in 'marital_status': ratio = 604.84",
        "Outcome disparity in 'marital_status': gap = 40.07%",
        "Proxy bias: 'relationship' correlates with 'sex' (r=0.580)",
        "Intersectional bias detected across marital_status x race",
      ],
    },
    dataset_comparison: {
      baseline_stats: {
        shape: [48842, 15],
        columns: [
          "age","workclass","fnlwgt","education","education_num",
          "marital_status","occupation","relationship","race","sex",
          "capital_gain","capital_loss","hours_per_week","native_country","class",
        ],
        dtypes: {
          age: "int64", workclass: "int32", fnlwgt: "int64",
          education: "int32", education_num: "int64",
          marital_status: "int32", occupation: "int32",
          relationship: "int32", race: "int32", sex: "int32",
          capital_gain: "int64", capital_loss: "int64",
          hours_per_week: "int64", native_country: "int32", class: "int32",
        },
        missing_values: 0,
        target_distribution: { "0": 37155, "1": 11687 },
        target_positive_rate: 0.2393,
      },
      debiased_stats: {
        shape: [156653, 15],
        columns: [
          "age","workclass","fnlwgt","education","education_num",
          "marital_status","occupation","relationship","race","sex",
          "capital_gain","capital_loss","hours_per_week","native_country","class",
        ],
        dtypes: {
          age: "int64", workclass: "int32", fnlwgt: "int64",
          education: "int32", education_num: "int64",
          marital_status: "int32", occupation: "int32",
          relationship: "int32", race: "int32", sex: "int32",
          capital_gain: "int64", capital_loss: "int64",
          hours_per_week: "int64", native_country: "int32", class: "int32",
        },
        missing_values: 0,
        target_distribution: { "0": 129409, "1": 27244 },
        target_positive_rate: 0.1739,
      },
      representation_shift: {
        sex: {
          groups: {
            Female: { before: 0.3315, after: 0.5329, delta: 0.2014 },
            Male: { before: 0.6685, after: 0.4671, delta: -0.2014 },
          },
          imbalance_before: 0.337,
          imbalance_after: 0.0658,
          imbalance_improved: true,
        },
        race: {
          groups: {
            White: { before: 0.855, after: 0.8161, delta: -0.039 },
            Black: { before: 0.0959, after: 0.127, delta: 0.031 },
            "Asian-Pac": { before: 0.0311, after: 0.0361, delta: 0.005 },
            "Amer-Indian": { before: 0.0096, after: 0.0109, delta: 0.0012 },
            Other: { before: 0.0083, after: 0.0101, delta: 0.0018 },
          },
          imbalance_before: 0.8467,
          imbalance_after: 0.806,
          imbalance_improved: true,
        },
      },
      outcome_rate_change: {
        sex: {
          groups: {
            Female: { before: 0.1093, after: 0.1258, delta: 0.0166 },
            Male: { before: 0.3038, after: 0.2288, delta: -0.075 },
          },
          disparity_before: 0.1945,
          disparity_after: 0.1029,
          disparity_improved: true,
        },
        race: {
          groups: {
            White: { before: 0.254, after: 0.1876, delta: -0.0663 },
            Black: { before: 0.1208, after: 0.0816, delta: -0.0392 },
            "Asian-Pac": { before: 0.2693, after: 0.2596, delta: -0.0097 },
            "Amer-Indian": { before: 0.117, after: 0.0412, delta: -0.0758 },
            Other: { before: 0.1232, after: 0.0614, delta: -0.0617 },
          },
          disparity_before: 0.1522,
          disparity_after: 0.2184,
          disparity_improved: false,
        },
      },
      fairness_deltas: {
        sex: {
          before: { dp_diff: 0.1945, di_ratio: 0.3597 },
          after: { dp_diff: 0.1029, di_ratio: 0.5497 },
          dp_diff_delta: -0.0916,
          di_ratio_delta: 0.19,
          dp_improved: true,
          di_improved: true,
        },
        race: {
          before: { dp_diff: 0.1522, di_ratio: 0.4346 },
          after: { dp_diff: 0.2184, di_ratio: 0.1588 },
          dp_diff_delta: 0.0662,
          di_ratio_delta: -0.2758,
          dp_improved: false,
          di_improved: false,
        },
      },
      statistical_tests: {
        age: { ks_statistic: 0.0261, p_value: 0.0, significantly_different: true },
        fnlwgt: { ks_statistic: 0.0114, p_value: 0.0001, significantly_different: true },
        education_num: { ks_statistic: 0.0182, p_value: 0.0, significantly_different: true },
        capital_gain: { ks_statistic: 0.0428, p_value: 0.0, significantly_different: true },
        hours_per_week: { ks_statistic: 0.0186, p_value: 0.0, significantly_different: true },
      },
      summary: {
        features_improved: ["sex"],
        features_worsened: ["race"],
        dataset_size_change: { before: 48842, after: 156653 },
        statistical_tests_run: 5,
        statistically_significant_changes: 5,
        overall_assessment: "Mitigation improved fairness",
      },
    },
    bias_tags: {
      representation_bias: true,
      outcome_bias: true,
      fairness_violation: true,
      proxy_bias: true,
      intersectional_bias: true,
    },
  },
  model_analysis: {
    model_output: {
      best_strategy: "resampling",
      accuracy_drop: -0.0485,
      comparison_table: [],
      metrics_before_after: {},
    },
    ranking: {
      best_strategy: "resampling",
      best_score: 0.5155,
      ranking_table: [
        { pipeline: "resampling", strategy: "resampling", rank: 1, score: 0.5155, accuracy: 0.8592, demographic_parity_diff: 0.0, disparate_impact: 1.0 },
        { pipeline: "smote", strategy: "smote", rank: 2, score: 0.5155, accuracy: 0.8592, demographic_parity_diff: 0.0, disparate_impact: 1.0 },
        { pipeline: "disparate_impact_remover + reweighting", strategy: "dir+reweighting", rank: 3, score: 0.4374, accuracy: 0.7854, demographic_parity_diff: 0.0846, disparate_impact: 0.4075 },
        { pipeline: "disparate_impact_remover + fairlearn", strategy: "dir+fairlearn", rank: 4, score: 0.4374, accuracy: 0.7854, demographic_parity_diff: 0.0846, disparate_impact: 0.4075 },
        { pipeline: "reweighting", strategy: "reweighting", rank: 5, score: 0.4256, accuracy: 0.8008, demographic_parity_diff: 0.1371, disparate_impact: 0.2155 },
        { pipeline: "fairlearn_reduction", strategy: "fairlearn", rank: 6, score: 0.4256, accuracy: 0.8008, demographic_parity_diff: 0.1371, disparate_impact: 0.2155 },
        { pipeline: "threshold_optimization", strategy: "threshold", rank: 7, score: 0.3986, accuracy: 0.8083, demographic_parity_diff: 0.2158, disparate_impact: 0.0 },
        { pipeline: "reweighting + threshold", strategy: "reweight+thresh", rank: 8, score: 0.3986, accuracy: 0.8083, demographic_parity_diff: 0.2158, disparate_impact: 0.0 },
        { pipeline: "baseline", strategy: "baseline", rank: 9, score: 0.3943, accuracy: 0.8107, demographic_parity_diff: 0.2303, disparate_impact: 0.0561 },
        { pipeline: "disparate_impact_remover", strategy: "dir", rank: 10, score: 0.3612, accuracy: 0.7853, demographic_parity_diff: 0.2751, disparate_impact: 0.0695 },
      ],
    },
    explainability: {
      feature_importance: {
        baseline: {
          marital_status: 0.2841,
          capital_gain: 0.2156,
          education_num: 0.1523,
          age: 0.1102,
          hours_per_week: 0.0891,
          relationship: 0.0654,
          occupation: 0.0412,
          capital_loss: 0.0221,
          sex: 0.0115,
          race: 0.0085,
        },
        best: {
          marital_status: 0.2612,
          capital_gain: 0.2298,
          education_num: 0.1687,
          age: 0.1245,
          hours_per_week: 0.0923,
          relationship: 0.0534,
          occupation: 0.0389,
          capital_loss: 0.0198,
          sex: 0.0072,
          race: 0.0042,
        },
      },
      model_comparison: {
        baseline_metrics: {
          accuracy: 0.8107,
          strategy: "baseline",
          rank: 9,
          demographic_parity_diff: 0.2303,
        },
        best_metrics: {
          accuracy: 0.8592,
          strategy: "resampling",
          rank: 1,
          demographic_parity_diff: 0.0,
        },
        strategies_evaluated: 10,
        comparison_table: [],
      },
      predictions_analysis: {
        baseline: { total: 9769, positive_count: 2345, negative_count: 7424, positive_rate: 0.2401 },
        best: { total: 9769, positive_count: 1802, negative_count: 7967, positive_rate: 0.1844 },
        prediction_agreement: 0.9123,
        predictions_changed: 857,
      },
      shap_summary: null,
      explanation:
        "The resampling strategy was applied to address representation imbalance across protected attributes. " +
        "By oversampling underrepresented groups, the model learned more equitable decision boundaries. " +
        "The top predictive features shifted from being dominated by marital_status (a proxy for gender bias) " +
        "to a more balanced distribution across education, capital gains, and age. " +
        "Notably, the importance of sensitive attributes (sex, race) decreased after mitigation, " +
        "indicating reduced reliance on protected characteristics for predictions.",
    },
  },
  explanations: {
    summary:
      "Representation bias, outcome bias, fairness violation, proxy bias, and intersectional bias were detected. " +
      "The resampling strategy improved fairness with a 4.85% increase in accuracy.",
    bias_explanation:
      "The UCI Adult dataset exhibits significant bias across multiple dimensions. " +
      "Gender bias is evident with males receiving positive outcomes (>$50K) at 30.4% vs 10.9% for females. " +
      "Racial disparities show White individuals at 25.4% positive outcomes compared to 12.1% for Black individuals. " +
      "The 'relationship' feature acts as a proxy for gender, showing 58% correlation with sex. " +
      "Intersectional analysis reveals compounded disadvantage: never-married Black individuals have only 2.2% positive outcome rate.",
    strategy_justification:
      "Resampling was selected as the optimal strategy because it directly addresses representation imbalance, " +
      "the root cause of bias in this dataset. By equalizing group sizes, the model no longer learns " +
      "discriminatory patterns from skewed training data. Unlike threshold optimization, which only adjusts " +
      "decision boundaries post-hoc, resampling fundamentally changes the training distribution.",
    tradeoff_analysis:
      "The accuracy-fairness tradeoff is highly favorable. The best model achieves 85.9% accuracy " +
      "(a 4.85% improvement over baseline) while reducing demographic parity difference from 0.23 to 0.00. " +
      "This is a rare case where both metrics improve simultaneously, as the balanced training data " +
      "helps the model generalize better across all subgroups.",
    recommendation: "Apply the 'resampling' mitigation strategy.",
    gemini_used: true,
  },
  metadata: {
    mode: "full_pipeline",
    strategy_used: "resampling",
    timestamp: "2026-04-23T19:24:07",
    elapsed_seconds: 29.2,
    config: {
      alpha: 0.6,
      beta: 0.4,
      model_type: "logistic_regression",
      gemini_enabled: true,
    },
  },
};
