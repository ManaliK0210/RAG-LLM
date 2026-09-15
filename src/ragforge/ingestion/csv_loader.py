from pathlib import Path

import pandas as pd

from ragforge.ingestion.document import Document


class CSVDocumentLoader:
    """
    Loader for CSV documents.

    Converts each CSV row into a readable text representation
    while preserving the original column names and source metadata.
    """

    def load(
        self,
        path: str | Path,
    ) -> Document:
        """
        Load a CSV document from disk.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                "CSVDocumentLoader only supports .csv files."
            )

        dataframe = pd.read_csv(path)

        if dataframe.empty:
            raise ValueError(
                f"CSV contains no rows: {path}"
            )

        rows = []

        for index, row in dataframe.iterrows():
            fields = []

            for column in dataframe.columns:
                value = row[column]

                if pd.isna(value):
                    value = ""

                fields.append(
                    f"{column}: {value}"
                )

            rows.append(
                f"Row {index + 1}:\n"
                + "\n".join(fields)
            )

        text = "\n\n".join(rows).strip()

        if not text:
            raise ValueError(
                f"CSV contains no extractable text: {path}"
            )

        return Document(
            text=text,
            source=str(path),
            metadata={
                "file_type": "csv",
                "row_count": len(dataframe),
                "column_count": len(dataframe.columns),
                "columns": list(
                    dataframe.columns
                ),
            },
        )