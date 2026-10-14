"""Clinical Guidelines RAG Assistant."""

import ssl


def _patch_windows_cert_loading() -> None:
    """Make Python tolerate one malformed certificate in the Windows store.

    Why: on some Windows machines a single broken certificate makes
    ``ssl.create_default_context()`` raise ``SSLError: nested asn1 error``,
    which stops libraries such as aiohttp (used by deepeval and Phoenix) from
    even importing. We load the certificates one by one and skip bad ones.
    Nothing here weakens verification: every valid certificate is still loaded.
    """
    if not hasattr(ssl.SSLContext, "_load_windows_store_certs"):
        return  # Not Windows, nothing to patch.

    def _load_windows_store_certs(self, storename, purpose):
        for cert, encoding, trust in ssl.enum_certificates(storename):
            if encoding != "x509_asn":
                continue
            if trust is not True and purpose.oid not in trust:
                continue
            try:
                self.load_verify_locations(cadata=cert)
            except ssl.SSLError:
                continue  # Skip only the malformed certificate.

    ssl.SSLContext._load_windows_store_certs = _load_windows_store_certs


_patch_windows_cert_loading()
