NVYRA-X SECURITY POLICY
EFFECTIVE DATE: December 28, 2025 STATUS: ACTIVE & ENFORCED

At nvyra-x, we regard the security of our reasoning engines, proprietary search algorithms, and model weights as paramount. Given the high-value nature of our intellectual property, we enforce a strict, zero-tolerance policy regarding the handling of security vulnerabilities.

1. REPORTING A VULNERABILITY
Do not file a GitHub Issue. Publicly disclosing a security vulnerability — even if well-intentioned - is a violation of our security policy and may result in the immediate revocation of access and potential legal action.

If you believe you have discovered a vulnerability in the nvyra-x stack, you must report it immediately and exclusively via:

Email: security@nvyra-x.com Subject Line: [CRITICAL] Security Finding - [Component Name]

We aim to acknowledge all reports within 24 hours.

2. In scope Vunerabilities 
We are specifically interested in reports covering the following vectors:

Model Exfiltration: Techniques that allow unauthorized extraction of model weights or proprietary KV cache states.
Inference Bypass: Methods to bypass the "Verifier/Critic" step to force the generation of unverified or unsafe outputs.
Prompt Injection / Jailbreaking: Adversarial inputs that cause the reasoning model to deviate from its system prompt or safety guardrails.
Side-Channel Leaks: Timing or resource-usage attacks against the inference engine that reveal details about the search tree depth or width.
Remote Code Execution (RCE): Any exploit allowing arbitrary code execution on the cluster controllers or inference nodes.

3. Out of Scope
Social engineering (phishing) of nvyra-x employees.
Physical security attacks against our data centers or GPU providers.
Denial of Service (DoS/DDoS) attacks that simply exhaust compute quotas.

4. Rules of Engagement
nvyra-x will not pursue legal action against you for security research that strictly adheres to these guidelines:

Private Disclosure: You must not disclose the vulnerability to any third party or the public until nvyra-x has explicitly authorized it in writing.
Non-Destructive: You must not harm our infrastructure, degrade the performance of our inference clusters, or destroy data.
Data Privacy: You must not view, access, or modify data belonging to other users or the company beyond what is strictly necessary to demonstrate the vulnerability.
No Financial Extortion: You must not demand payment in exchange for withholding the vulnerability disclosure.

6. Encrypted Communication
For highly sensitive reports (e.g., involving zero-day exploits in the inference kernel), please encrypt your email using our PGP key:


-----BEGIN PGP PUBLIC KEY BLOCK-----
mQINBGlReIUBEADXA6MNv1PNT2Y2kLgmdRzCg8n5PKd8raInM6wWAF6R/QV8XeS4
jQLAWoHkbuxhUVxodCzIY1pKfFI8C30HY8/pbhiqjkAMAQ6J03oG1UZ68Fc9AX05
B5liNuzm3rXHeJrBcWHAf9CCxOSJuOLO9bc8UCi1z48qL2q24dCf/rWV+KbtspU2
658VAAsCJOWmuaEBWDezeYSIimaoQ3gE5Iode7I2fNbuUuYzfh1VWvmMqxYEcdjZ
Wur2+me+mtqur9mZOXDjJX6WEcnp8ejN7xRRB9xpcfLQv0a04T7mN3IIQQEysxeu
JcfbKmGOZpK7YJGb2BYX8gpOpHqVVHA1t15YXSkKrsJLL+67Wq/snITo9+JAkZGK
lb0nqw9sQeY85xaIGpm1gCntdG+4vOhYDRLvrGxIVF+oYS2sSN+zKEXybIz+LqtI
m7WGc9F7SbO6IawGrqqbemxC1pm7fQXlkXW8i3YatW+UygQDVD4+abkptQh3qclc
eOsVSkgBnY1+Mbsv1DgvXMwzULZKPBzKNGbHKBlKTvKNJsmorM9req2HsDHBCTgE
jwLnI6RZrNMZXn2CFd1aKw6DYlqlJxuEtTo3rgXh1HZGI54e8yZKqzO3zhFkvKuv
BfNirHeqffZYDVOmqW6Hv+ZhMisJCUZG7jHCde2mG/LQfrLLvkGasdG5FwARAQAB
tB5udnlyYS14IDxzZWN1cml0eUBudnlyYS14LmNvbT6JAlEEEwEIADsWIQTL6wtE
t76lPR4w2dKW/w++HycTPQUCaVF4hQIbAwULCQgHAgIiAgYVCgkICwIEFgIDAQIe
BwIXgAAKCRCW/w++HycTPbhmD/wLs3DeonBxvik/h23NMheLcplOVkjSoVNF5tTv
4HEoffzRbrzofOm1ttLLxr1BL06d0JmtT15uX/A4UDfVGsXNZsDiCuFiOxsVkPrz
ViJbZRdvlo7TgBMRUEQ5W9UeNbvlgj4RkZAf9WH7qrzvzeGXUWsSRWkmMOKCMw6z
Zs0TUxwmdODPLEI2NOGiOH4Mcc1TRQClQzDlvwzRyj4r/WhkfWjHhUavGAuMjQCN
uZReQQrcPPHS+6Mb9puYVzyRSARDFEvfHzabTg6hg8XFQxPlZZp4IlratGyene5Y
ROqP9DlvayEvyMzH568l+iqjMvTgf0IlLBFehqQBzcq49znz/dOYkdV6FAnbrODs
X+RtUD8zzHndMEN542sV4uFThZwlIpeFM9BqD8o6j4X9pAf63C83CgbW6eJAZaHR
sZJDghd676HesDBb9ABO4IXWT6lMVOmfPNdTuYkmrWqDrmTa4C0SIz+rfz6n2s7f
uEXRszVrP4IRzCtLLJYNyaUoe4B0vgBkFdRMxdBpltolgKv9ne+76wIiIikXWsL6
sJVhA+X2EYy7R9VBghguvbYY7vaWbQvXrDqX4Va0wxhxN/l/LXA6n7MoBtaLWVOR
pCjmZ7TTRlrb4zJBB7tJaHi1hyrBwFhDeNAKjL7NXWur0MpWExGUbUoUZK9lvASP
flH0V7kCDQRpUXiFARAAosT16d40Pn/49DImtHuB00yJMwAM4+Cnqkw/e48nOUMm
P4cOsPJh/QRttiCCf3/wPk3+qA5S7rhDndKsQWWYA8tsd1bP6ctzEHnjSlNkkY3z
T+EF4eSZzt7RIi5P5yzl6/yCmbjgQ1/ObzB7dwXBmcHmhNEaHlDX4rrENrrGUQWA
lCJxNJu7knno/VasVT9lgvcfCFnZiOlbqMghkcHXf71Vt36CBOPbfZqVoP4EZpiq
qv6O5KgfKRLF8tHykMd14/H1xaKUxLSCEv8lt471kWsmH3snMjLqRHIqbNtYBMVY
CUtd09GUp/QjajzZ7O1k3x6xfdALbWgeyHjLJ6q4ng0mM2FmbEa6xZ0dN1h2kmfn
qVkpH7ifD2PtUy61pePxKZ/rhnMy/Ns4BCtTIu65OmAQAxBERJez26HS1914zerL
xjhYwyZEoblffNmZdU/zRXlG9OMS+oCtPZ865O7FO/fQLs8Blt3kIH3ht6/6h+bO
P3h372v8Ny6+IObm7fEO0RmhuvNQh6DNeXqWRjH6upk6DzgwKOTg0Wk/NCDxA+6d
KPXxto1s/SZJKGo+6oQHD6KgOqmjSuHvmgGZh5eO/YN5T7KtEI62HMfGzmZvUMNw
JBRZ1zG+tO9S3/QLctHsLLYoIT8m8pERdoV5byDQl6Wt1QUidoYxoVuR2fVx2PcA
EQEAAYkCNgQYAQgAIBYhBMvrC0S3vqU9HjDZ0pb/D74fJxM9BQJpUXiFAhsMAAoJ
EJb/D74fJxM9/oYQAKXEbCA0DKdh/abAsq2CWIyZ1m3THHO+5AHZE7/OQHwNVOxs
8yglml8KQKisYqX9za8B7N1lJbVPv+DQokaC8nRNywgSbl1xMyLrF8lH803S+HCo
tfIWaQTZhNR2s8vnzWWhd/6bzUVulGexfVgTMvlpBRNZGpWZsxE+/iWVz7w9Nppv
Zgzi8gsIJ07bHw0rzVkYt2MRWCaZqiYpJbdjSVEfa8LFNO6FFdrR9EDVpIsz1GCt
3IUsHDgVvAgAYQbILNwTTV5nEHj+WqN/+L8B56J2QeTH2NCuMz3A7SuJKoRISNw4
fbwxOsCQcc+aZZW6dJFT3NXPnSYbCa2gUvduoJNe9VLkzRo0wB/p2nC7Oly74gvc
DWMTrEfNXAEEL9HZYffQIHDFotQmr2qxXjHJvDYBOULvQTmBXHKaSUutm616WdDO
snZeXz1uoPIm1H6e50JWh90ijrW8tZuYgdy3US+midLPAr7FmHn7fcMm0SHr8IjR
YnqMZF7esG3hocwWaYK9rv5UgleApDODm9z+xHtFchVzpajyRyOJwwQlHLxLlBdg
p2uD/Qg+d99p/yoxtuXDYdS9uRPPOgqfmGybMT8J2d9ippNNuMDlR1W644IuB/wY
mvovlBIpqnAuOCZrcLKVAXLSvhbogJVaK/sx7JNIBQCU3xSp+YRyWGShpMAP
=uAbO
-----END PGP PUBLIC KEY BLOCK-----
6. BOUNTY & RECOGNITION
nvyra-x does not operate a public bug bounty program at this time. However, we may, at our sole discretion, offer financial rewards or recognition for critical findings that prevent significant loss of IP or compute resources.
