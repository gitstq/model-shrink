# 🤝 Contributing to ModelShrink

Thanks for taking the time to contribute! ModelShrink is meant to stay small,
readable and dependency-light, and good contributions are very welcome.

## 🧭 Getting started

1. Fork the repository and clone your fork.
2. Set up an editable install with development extras:

   ```bash
   pip install -e .
   pip install matplotlib   # optional, for `--plots`
   ```

3. Run the demo to make sure everything works:

   ```bash
   modelshrink demo
   ```

## ✅ Running the tests

```bash
make test
# or
python3 -m unittest discover -s tests -v
```

Please add tests for new behavior and make sure the full suite passes before
opening a Pull Request.

## 📐 Code style

- Keep it **readable and explicit**; prefer a few clear lines over clever code.
- Keep dependencies light — the core runtime should stay NumPy-only.
- Add docstrings for public functions/classes and comments where the math is
  non-obvious.
- Follow the surrounding style of the file.

## 💬 Commit & PR conventions

We use the [Angular / Conventional Commits](https://www.conventionalcommits.org/)
style:

- `feat: add a new feature`
- `fix: correct a bug`
- `docs: documentation changes`
- `refactor: code restructuring without behavior change`
- `test: adding or adjusting tests`
- `chore: tooling / build changes`

## 🔀 Pull Request process

1. Create a focused branch (e.g. `feat/int-packing`).
2. Make logical commits and keep the change scoped.
3. Ensure tests pass and update docs/examples if needed.
4. Open the PR and describe **what** changes and **why**.

## 🐛 Reporting issues

Open an issue with a clear title, steps to reproduce, expected vs actual
behavior, and your Python/NumPy versions.

By contributing, you agree your contributions are licensed under the project's
MIT License. Happy shrinking! 🗜️
