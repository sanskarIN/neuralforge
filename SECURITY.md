# Security Policy

## Reporting a security issue

Please do not open a public issue containing credentials, API keys, private user data, exploit secrets, or other sensitive information.

For ordinary bugs, documentation problems, reproducibility issues, dependency concerns, or unsafe defaults that do not expose private secrets, open a GitHub issue with a minimal reproducible example.

## Repository safety rules

- Never commit real API keys, passwords, access tokens, cloud credentials, private certificates, or identity documents.
- Use environment variables and local `.env` files for secrets; `.env` files are ignored by the repository.
- Use synthetic or properly licensed datasets in examples whenever possible.
- Pin or document dependency versions for reproducible examples.
- Review model and dataset licenses before redistributing assets.
- Do not treat educational examples as production-ready security controls without review.

Storefront/payment issues are separate from repository security reports. The official Ram Sandesh storefront is **https://ramsandesh.gumroad.com**.
