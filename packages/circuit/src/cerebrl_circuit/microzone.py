"""Single cerebellar microzone assembly metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MicrozoneScale:
    granule: int
    golgi: int
    purkinje: int
    mli: int
    dcn_projection: int
    dcn_inhibitory: int
    inferior_olive: int
    m1_l5: int
    pontine: int
    vl: int

    @property
    def total(self) -> int:
        return (
            self.granule
            + self.golgi
            + self.purkinje
            + self.mli
            + self.dcn_projection
            + self.dcn_inhibitory
            + self.inferior_olive
            + self.m1_l5
            + self.pontine
            + self.vl
        )


def minimal_microzone() -> MicrozoneScale:
    return MicrozoneScale(
        granule=5000,
        golgi=50,
        purkinje=32,
        mli=20,
        dcn_projection=10,
        dcn_inhibitory=5,
        inferior_olive=4,
        m1_l5=100,
        pontine=50,
        vl=20,
    )

