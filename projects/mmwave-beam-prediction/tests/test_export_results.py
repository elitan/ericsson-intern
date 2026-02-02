from beampred.export_results import _fmt, _fmt_txt


class TestFmt:
    def test_with_std(self):
        result = _fmt(0.95, 0.01)
        assert "0.950" in result
        assert "0.010" in result

    def test_without_std(self):
        result = _fmt(0.95, 0)
        assert result == "0.950"


class TestFmtTxt:
    def test_with_std(self):
        result = _fmt_txt(0.95, 0.01)
        assert "±" in result

    def test_without_std(self):
        result = _fmt_txt(0.95, 0)
        assert result == "0.9500"
