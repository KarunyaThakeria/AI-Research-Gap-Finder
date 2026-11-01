import re


def clean_text(text):

    # Remove extra spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove excessive blank lines
    text = re.sub(r'\n+', '\n', text)

    # Remove page numbers appearing alone
    text = re.sub(
        r'(?m)^\s*\d+\s*$',
        '',
        text
    )

    # Remove unwanted control characters
    text = re.sub(
        r'[\x00-\x08\x0b\x0c\x0e-\x1f]',
        '',
        text
    )

    # Fix hyphenation across line breaks
    text = re.sub(
        r'(\w+)-\n(\w+)',
        r'\1\2',
        text
    )

    # Normalize spaces around newlines
    text = re.sub(
        r' *\n *',
        '\n',
        text
    )

    return text.strip()