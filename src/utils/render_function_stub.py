import textwrap


def render_function_stub(is_async, name: str, signature: str, docstring: str|None) -> str:
    """
    Render a function's public contract as a prompt-ready function stub.
    """

    declaration = "async def" if is_async else "def"
    header = (
        f"{declaration} "
        f"{name}"
        f"{signature}:"
    )

    if docstring:
        # Escaping triple quotes is usually enough for prompt construction.
        escaped_docstring = docstring.replace('"""', r'\"\"\"')
        body = f'"""\n{escaped_docstring}\n"""\n...'
    else:
        body = '"""No docstring provided."""\n...'

    return f"{header}\n{textwrap.indent(body, '    ')}"