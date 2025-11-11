# Secure Code Generation with Mistral

This repository contains a Streamlit application that leverages Mistral language models to
produce secure coding recommendations. Users can provide natural-language instructions,
attach dependency manifest files, and include security scan outputs to obtain tailored guidance
on remediation steps and safe-by-default code patterns.

## Features

- 📄 **Manifest Upload** – ingest dependency descriptors such as `package.json`,
  `requirements.txt`, or `pyproject.toml` for supply chain analysis context.
- 📝 **Security Scan Upload** – attach results from SAST/DAST/dependency tooling to focus the
  remediation guidance.
- 🎚️ **Security Focus Controls** – toggle additional review passes for dependency analysis,
  static code inspection, and secret detection.
- 🤖 **Mistral Integration** – query hosted Mistral chat models with configurable temperature to
  balance determinism and creativity.
- 📥 **Downloadable Reports** – export generated recommendations as a Markdown document.

## Getting Started

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Export your Mistral API key so the application can invoke the hosted models:

   ```bash
   export MISTRAL_API_KEY="your_api_key_here"
   ```

3. Launch the Streamlit application:

   ```bash
   streamlit run app.py
   ```

4. Open the provided local URL, supply instructions, and optionally upload manifest and security
   scan files to generate secure code recommendations.

> **Security Tip:** Avoid uploading sensitive or proprietary information. Review and vet any
> generated suggestions before applying them to production systems.
