# Public release checklist

- Select a software license approved by all authors and institutions.
- Replace draft citation metadata with the final venue, DOI, and URL.
- Confirm author names and preferred contact address.
- Add the final paper PDF link when public.
- Verify dataset download instructions against each dataset's current license.
- Record immutable model and dataset revisions for each reported table.
- Re-run the reviewer smoke test and unit tests on a clean clone.
- Run a small CPU or GPU Hugging Face example with the documented command.
- Inspect git status and confirm there are no checkpoints, archives, logs,
  personal paths, credentials, or private data.
- Create a tagged release after the final commit.

