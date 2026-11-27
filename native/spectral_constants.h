/* SPDX-License-Identifier: GPL-2.0-or-later */
/* DD coefficients of sin(x)/x on [0, pi/2], k=0..19.
 * Exact rational Taylor coefficients split into nearest binary64 limbs.
 * pi/2 split from a 140-decimal-digit reference. */
static const dd_real dd_pi_half = { 0x1.921fb54442d18p+0, 0x1.1a62633145c07p-54 };
static const dd_real dd_sine_coefficients[20] = {
    { 0x1.0000000000000p+0, 0x0.0p+0 },
    { -0x1.5555555555555p-3, -0x1.5555555555555p-57 },
    { 0x1.1111111111111p-7, 0x1.1111111111111p-63 },
    { -0x1.a01a01a01a01ap-13, -0x1.a01a01a01a01ap-73 },
    { 0x1.71de3a556c734p-19, -0x1.c154f8ddc6c00p-73 },
    { -0x1.ae64567f544e4p-26, 0x1.c062e06d1f209p-80 },
    { 0x1.6124613a86d09p-33, 0x1.f28e0cc748ebep-87 },
    { -0x1.ae7f3e733b81fp-41, -0x1.1d8656b0ee8cbp-97 },
    { 0x1.952c77030ad4ap-49, 0x1.ac981465ddc6cp-103 },
    { -0x1.2f49b46814157p-57, -0x1.2650f61dbdcb4p-112 },
    { 0x1.71b8ef6dcf572p-66, -0x1.d043ae40c4647p-120 },
    { -0x1.761b41316381ap-75, 0x1.3423c7d91404fp-130 },
    { 0x1.3f3ccdd165fa9p-84, -0x1.58ddadf344487p-139 },
    { -0x1.d1ab1c2dccea3p-94, -0x1.054d0c78aea14p-149 },
    { 0x1.259f98b4358adp-103, 0x1.eaf8c39dd9bc5p-157 },
    { -0x1.434d2e783f5bcp-113, -0x1.0b87b91be9affp-167 },
    { 0x1.3981254dd0d52p-123, -0x1.2b1f4c8015a2fp-177 },
    { -0x1.0dc59c716d91fp-133, -0x1.419e3fad3f031p-188 },
    { 0x1.9ec8d1c94e85bp-144, -0x1.670e9d4784ec6p-201 },
    { -0x1.1e99449a4bacep-154, 0x1.fefbb89514b3cp-210 },
};
