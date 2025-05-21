import html2text

class CustomHTML2Text(html2text.HTML2Text):
    """
    # CustomHTML2Text
    CustomHTML2Text is a subclass of html2text.HTML2Text that overrides some of its default behavior. It is used to convert HTML content to plain text in Python.

    ## Attributes
    ignore_links: A boolean indicating whether links should be ignored during conversion. Default is True.
    ignore_images: A boolean indicating whether images should be ignored during conversion. Default is True.
    ignore_emphasis: A boolean indicating whether emphasis (bold, italic, etc.) should be ignored during conversion. Default is True.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_links = True
        self.ignore_images = True
        self.ignore_emphasis = True
        # self.ignore_tables = True

    def handle_starttag(self, tag, attrs):
        if tag.lower().startswith("h"):
            pass
        else:
            super().handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag.lower().startswith("h"):
            pass
        else:
            super().handle_endtag(tag)