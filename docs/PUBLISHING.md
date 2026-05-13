# Publishing Checklist

1. Confirm the repository license before publishing. No license has been added
   automatically.
2. Review `figures/paper/` and `data/simulation_outputs/` for files that should
   or should not be public.
3. Run `python scripts/collect_paper_figures.py` to assemble a disposable figure
   bundle under `outputs/paper_figures/`.
4. Commit the organized repository.
5. Create a public remote and push:

```sh
gh repo create ai-regulation-games --public --source=. --remote=origin --push
```

