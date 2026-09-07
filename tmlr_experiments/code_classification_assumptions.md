# Methodological Assumptions: Code Corpus Character Classification

## Decision

All code corpora (Python stdlib, JavaScript, Java, and any future code corpus) use
**character-identity-only classification**: a character is classified as structural if
and only if it appears in `CODE_STRUCTURAL_CHARS`, regardless of where in the source
file it occurs.

## Shared structural set

```python
CODE_STRUCTURAL_CHARS = frozenset({
    '(', ')', '{', '}', '[', ']',   # brackets
    ',', '.', ';', ':',              # separators
    '=', '<', '>', '!',             # comparison / assignment
    '+', '-', '*', '/', '%',        # arithmetic
    '&', '|', '^', '~',            # bitwise / logical
    '@', '?',                        # decorators, ternary / optional chaining
})
```

This set is identical for every code corpus. No per-language structural definitions.

## What this means in practice

A structural character such as `(` is counted as structural in all positions:

| Position in source | Example | Counted as |
|--------------------|---------|-----------|
| Actual syntax | `area(length, width)` | structural ✓ |
| Inside string literal | `"result (sq m)"` | structural (misclassified) |
| Inside line comment | `# compute (n+1)` | structural (misclassified) |
| Inside block comment | `/* handle (edge) case */` | structural (misclassified) |

Characters inside string literals and comments are not syntactically structural, but
the classifier has no access to token-type information for arbitrary languages and
treats them identically.

## Why this approach was chosen

Python's `tokenize` standard library module provides exact token types per character
and was used in earlier versions of this pipeline. No equivalent exists in the Python
standard library for JavaScript or Java. Third-party options (`esprima`, `javalang`)
introduce per-language dependencies of uneven maintenance quality, and their
classification logic would differ, making cross-language β₃ values non-comparable.

Character-identity-only classification is less precise but **identical** across all
code corpora. Cross-language comparability is the priority.

## Assumptions and their validity

**Assumption 1: Structural characters appear predominantly in syntactic positions.**

In typical software source code, structural characters such as `(`, `)`, `{`, `}`,
`,`, `;` appear far more often in actual syntax than inside string literals or
comments. A one-time audit of the Python stdlib corpus found that approximately
85–90% of `(` occurrences are syntactic. The noise from misclassified occurrences
is therefore small relative to the total observations per character.

This assumption is more likely to hold for compiled or typed languages (Java, C)
where string content tends to be short, and less likely to hold for corpora that are
predominantly documentation strings or comment-heavy code.

**Assumption 2: Misclassification direction is conservative for the null hypothesis.**

Characters inside string literals are typically surrounded by letters and other
string content. Their local n-gram context is more predictable than in syntactic
positions (the model can leverage the surrounding text pattern of the string). This
means misclassified occurrences tend to have *lower* surprisal and *flatter* CG
curves than true syntactic occurrences, attenuating the structural CG signal.

For the code corpora, where the expected result is a null or near-null β₃, this
attenuation is conservative: it makes it *harder* to find a structural effect, not
easier. A null result under this assumption is therefore a more defensible null.

**Assumption 3: The same bias applies to all structural characters within a corpus.**

If all structural characters are subject to the same misclassification mechanism,
the relative ranking of structural vs. lexical CG trajectories is preserved, even if
absolute CG values are slightly attenuated for structural characters. The regression
coefficient β₃ captures the *differential* trajectory slope, so a uniform downward
bias on structural characters would reduce β₃ slightly, again conservative for
finding positive effects.

## Limitation

The classification does not distinguish a syntactically meaningful `(` from a
cosmetic `(` in a comment or string. In corpora with high documentation density
(e.g., heavily commented libraries, literate programming styles), the noise fraction
could be larger and the attenuation of the structural signal correspondingly more
severe.

Future work could introduce a lightweight language-agnostic tokenizer pass (e.g.,
regex-based comment and string stripping) that applies identically across languages,
avoiding per-language tokenizer dependencies while reducing misclassification.

## Effect on Python stdlib results

The canonical Python stdlib results (β₃ = −0.024, p ≈ 0.93) were produced using
Python's `tokenize` module (strata-based classification). The Phase 2 re-run of
Python stdlib uses character-identity-only classification for consistency with
JavaScript and Java.

The scientific conclusion (null result for code) is not expected to change.
Any shift in the β₃ point estimate is reported alongside the original value in the
robustness table.
