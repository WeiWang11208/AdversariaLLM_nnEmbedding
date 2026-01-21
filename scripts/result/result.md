| Attack                    | Dataset         | Total Samples | Success |   ASR   | Avg Loss | Min Loss | Avg Suffix Tokens | Avg Suffix PPL | Avg Time/Sample |
|---------------------------|-----------------|---------------|---------|---------|----------|----------|-------------------|----------------|-----------------|
| beast                     | adv_behaviors   | 300           | 175     |  58.3%  | 1.7510   | 0.3844   |        N/A        |      N/A       | 112.52s         |
| gcg                       | adv_behaviors   | 300           | 280     |  93.3%  | 1.6910   | 0.3906   |        N/A        |      N/A       | 341.84s         |
| pgd                       | adv_behaviors   | 300           | 300     | 100.0%  | 1.9540   | 0.4964   |        N/A        |      N/A       | 33.12s          |
| natural_suffix_embedding   | adv_behaviors   | 300           | 296     |  98.7%  | 9.3535   | 7.2506   |      21.7         |    32.84       | 32.85s          |
| gcg                       | strong_reject   | 313           | 230     |  73.5%  | 2.8406   | 0.6836   |        N/A        |      N/A       | 302.12s         |
| beast                     | strong_reject   | 313           | 79      |  25.2%  | 2.9715   | 0.7611   |        N/A        |      N/A       | 85.99s          |
| pgd                       | strong_reject   | 313           | 313     | 100.0%  | 3.0568   | 0.9097   |        N/A        |      N/A       | 29.67s          |
| natural_suffix_embedding   | strong_reject   | 313           | 311     |  99.4%  | 9.9146   | 7.5058   |        N/A        |      N/A       | 29.35s          