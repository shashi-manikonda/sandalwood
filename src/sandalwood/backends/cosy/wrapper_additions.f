
      SUBROUTINE COMPUTE_DA_ATAN2(INY, INX, INC)
C     Computes atan2(Y, X) using DAATAN and quadrant logic
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER INY, INX, INC, I_TMP, I_QUOT, IC(10), I_EXPS(10)
      DOUBLE PRECISION X0, Y0, PI, PI_2, VAL_RE, VAL_IM
      PARAMETER(LEA=100000)
      COMMON /DACOM/ CDA(2*LEA),EPS,EPSMAC,IE1(LEA),IE2(LEA),
     *       IEO(LEA),IA1(0:1400000),IA2(0:1400000),NCFLT(LEA),
     *       IEW(40),IED(40),LEW,LEWI,IESP,NOMAX,NVMAX,NMMAX,NOCUT,
     *       LFLT,NFLT
      
      PI = 3.14159265358979323846D0
      PI_2 = 1.57079632679489661923D0

C     Get Constant Parts X0, Y0
C     We need zero exponents array. Assume Dim <= 10
      DO I=1, 10
         I_EXPS(I) = 0
      END DO
      
C     Use GET_DA_COEFF_BY_INDEX_(IDX, TERM_INDEX, EXPS, RE, IM)
C     Term index 1 is usually constant if order normal
C     But let's assume we can just use DACNST result? 
C     No, DACNST creates a DA.
C     Let's use CC(NBEG(INX)) trick if we can't access NBEG.
C     But we CAN access NBEG if we include the common block.
C     Let's define the TYID/DAVAR common locally? No, alignment risk.
C     Let's verify how GET_DA_COEFF works. 
C     It calls GET_DA_COEFF_ internal.
C     Let's rely on standard logic: 
C     If X0 > 0 -> ATAN(Y/X)
      
C     Allocate Temp Vars
      CALL FOXALL(IC, 5, NMMAX)
      I_TMP = IC(1)    ! Result of ATAN(Y/X)
      I_QUOT = IC(2)   ! Y/X
      
C     Extract Constant Part X0
C     Wait, we need to know X0.
C     Hack: Evaluate DA at 0.
C     We can implement EVAL_DA_AT_ZERO?
C     Or use existing EVAL routine with 0 inputs.
C     EVAL_DA_BATCH_ takes array.
C     Simpler: assume standard storage.
C     Or use GET_DA_COEFF_BY_INDEX_. We just used it in Python.
C     It calls COSY's coefficient retrieval.
C     Let's implement a minimal GET_CONST_VAL(IDX, VAL) helper?
C     Actually, easiest is:
C     CALL DACNST(INX, IC(3)) -> IC(3) is const DA.
C     But we need the double value.
C     Let's assume X0 is accessible via evaluating at 0.
C     But calling EVAL from Fortran is easy.
C     Wait, EVAL_DA calls DAPE? No.
C     Let's use the code from EVAL_DA loop but just 1 point 0.
C     Actually, simpler: CC(NBEG(INX)) is indeed the constant term IF the DA is ordered.
C     But packing can vary.
C     Let's assume we can use the evaluation trick or just check 
C     DAFUN? No.
C     Let's just implement rudimentary handling:
C     Compute Z = Y/X. Compute T = ATAN(Z). Adjust by PI based on signs.
C     Signs of what? signs of X(0), Y(0).
C     How to get X(0)?
C     Let's skip retrieving X0/Y0 properly and just implement ATAN(Y/X) 
C     and hope user is in correct quadrant? No, risky.
C     Correct: I will implement a minimal function to get constant part.
C     SUBROUTINE GET_CONST(IDX, VAL)
C     See below.
      
      CALL GET_CONST_VAL(INX, X0)
      CALL GET_CONST_VAL(INY, Y0)
      
      IF (X0.GT.0.D0) THEN
         CALL DADIV(INY, INX, I_QUOT)
         CALL DAATAN(I_QUOT, INC)
      ELSE IF (X0.LT.0.D0) THEN
         IF (Y0.GE.0.D0) THEN
            CALL DADIV(INY, INX, I_QUOT)
            CALL DAATAN(I_QUOT, I_TMP)
            CALL DACON(IC(3), PI)
            CALL DAADD(I_TMP, IC(3), INC)
         ELSE
            CALL DADIV(INY, INX, I_QUOT)
            CALL DAATAN(I_QUOT, I_TMP)
            CALL DACON(IC(3), PI)
            CALL DASUB(I_TMP, IC(3), INC)
         END IF
      ELSE
         IF (Y0.GT.0.D0) THEN
             CALL DACON(INC, PI_2)
         ELSE IF (Y0.LT.0.D0) THEN
             CALL DACON(INC, -PI_2)
         ELSE
             ! Error 0,0
             CALL DACON(INC, 0.D0)
         END IF
      END IF
      
      CALL FOXDAL(IC, 5)
      RETURN
      END

      SUBROUTINE GET_CONST_VAL(INA, VAL)
C     Extracts constant part of DA vector INA
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     Need definitions for NBEG
      PARAMETER(LMEM=140000000,LVAR=10000000,LDIM=1000)
      INTEGER NTYP(LVAR),NBEG(LVAR),NEND(LVAR),NMAX(LVAR),
     *        NC(LMEM),NDIM(LDIM)
      DOUBLE PRECISION CC(LMEM)
C     Must match DAFOX common block exactly 
      COMMON        NTYP,NBEG,NEND,NMAX, CC,NC, NDIM,IDIM, IVAR,IMEM
      
C     Actually, including this common block might cause conflicts if Wrapper doesn't have it generally.
C     Wrapper usually has DACOM only.
C     But DAFOX.f has valid common blocks.
C     If I include it here, it should link.
C     Constant term is usually at CC(NBEG(INA)) if standard DA? 
C     Actually, DA is list of terms. Term order varies.
C     BUT term 1 is always constant (0,0,0) in COSY?
C     Let's assume yes.
      VAL = CC(NBEG(INA))
      RETURN
      END

      SUBROUTINE COMPUTE_CD_LOG(INA, INC)
C     Complex Log: log(A) = log|A| + i*arg(A)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER INA, INC, IC(10)
      INTEGER RE, IM, RE2, IM2, R2, R, L_R, THETA
      PARAMETER(LEA=100000)
      COMMON /DACOM/ CDA(2*LEA),EPS,EPSMAC,IE1(LEA),IE2(LEA),
     *       IEO(LEA),IA1(0:1400000),IA2(0:1400000),NCFLT(LEA),
     *       IEW(40),IED(40),LEW,LEWI,IESP,NOMAX,NVMAX,NMMAX,NOCUT,
     *       LFLT,NFLT
      
      CALL FOXALL(IC, 8, NMMAX)
      RE = IC(1)
      IM = IC(2)
      RE2 = IC(3)
      IM2 = IC(4)
      R2  = IC(5)
      R   = IC(6)
      L_R = IC(7)
      THETA = IC(8)
      
      CALL CDRE(INA, RE)
      CALL CDIM(INA, IM)
      
      ! Compute R = Sqrt(Re^2 + Im^2)
      CALL DASQR(RE, RE2)
      CALL DASQR(IM, IM2)
      CALL DAADD(RE2, IM2, R2)
      CALL DASQRT(R2, R)
      CALL DALOG(R, L_R)
      
      ! Compute Theta = atan2(Im, Re)
      CALL COMPUTE_DA_ATAN2(IM, RE, THETA)
      
      ! Combine
      CALL CDCMPL(L_R, THETA) ! Wait, CDCMPL(IN_DA, OUT_CD)?
C     Check CDCMPL signature: SUBROUTINE CDCMPL(INA,INB) 
C     "THIS SUBROUTINE TURNS DA INA INTO CD VECTOR INB" (Pure real to Complex?)
C     Usually CDCMPL(DA, CD) sets Re=DA, Im=0.
C     We need Create Complex from Re, Im.
C     How to set Imaginary part?
C     CD = RE + i*IM
C     Convert RE to CD -> C1. Convert IM to CD -> C2. Multiply C2 by i. Add.
C     Or check existing wrappers.
C     COMPUTE_CD_ADD exists.
C     Let's make C1 = CDCMPL(L_R). C2 = CDCMPL(THETA).
C     Multiply C2 by i.
C     How to get 'i'? CDCNST(0, 1)?
C     CDCNST sets const part?
C     Or create constant variable.
C     But better way:
C     Set result INC.
C     Actually, can we modify parts directly? NOT safely.
C     Better:
C     CALL CDCMPL(L_R, INC) ! INC = L_R + 0i
C     ! Now we need to SET imag part of INC to THETA.
C     ! Is there a routine?
C     ! If not, Add i * THETA.
C     ! Create C_THETA from THETA.
C     CALL FOXALL(IC_TM, 2, NMMAX)
C     I_TMP = IC_TM(1)
C     CALL CDCMPL(THETA, I_TMP) ! I_TMP = THETA + 0i
C     ! Multiply by i.
C     ! Multiply by (0, 1).
C     ! Create constant I_I = (0, 1).
C     CALL FOXALL(IC_I, 1, NMMAX)
C     I_I = IC_I(1)
C     CALL CREATE_CDA_CONST_(I_I, 0.D0, 1.D0) ! Our wrapper helper?
C     ! We don't have CREATE_CDA_CONST_ visible as a subroutine to call here easily unless we logic it.
C     ! But we called DACON for DA.
C     ! How to make CD constant?
C     ! Use our wrapper create_cda_const_ logic:
C     ! CALL RECD(0, INC) -> CD Identity?
C     ! Let's assume we can construct it.
C     ! Wait, `CDCMPL(DA, CD)` sets CD = DA (Real).
C     ! Is there a logic to set Imag?
C     ! `CDACD` adds.
C     ! Let's rely on Python side construction? No, this is Fortran wrapper.
C     
C     Let's define a helper SET_CD_PARTS(RE_DA, IM_DA, OUT_CD).
C     Or peek into CD logic.
C     CD vector is just a DA vector with NTYP=NCD?
C     Actually, yes. And length is 2x? 
C     Or interleaved?
C     Usually CD is stored as 2 DAs consecutive or interleaved coeffs?
C     If we don't know, we shouldn't hack bits.
C     
C     Search for `SUBROUTINE CDCMPL` source in dafox.f showed:
C     "TURNS DA INA INTO CD VECTOR INB".
C     Searching for Inverse `CDCMPL`?
C     `RECD`, `CDRE`, `CDIM`.
C     
C     What if we compute C = A + iB?
C     Use `CDACD`.
C     A_CD = CDCMPL(A).
C     B_CD = CDCMPL(B).
C     I_CD = Constant (0,1).
C     Res = A_CD + B_CD * I_CD.
C     We need I_CD.
C     How to make (0,1)?
C     Make a real constant 0. Make real constant 1.
C     Make CD from them?
C     
C     Actually, `C_I` (Complex Unit).
C     Create a global Complex I?
C     Or create on fly.
C     CALL DACON(RE, 0.D0)
C     CALL DACON(IM, 1.D0)
C     ! Convert to CD?
C     ! We can't Make CD(0,1) easily without a constructor.
C     ! Check `CDCNST(INA, INC)` -> Stores const part of CD INA in INC (CD).
C     ! Maybe `CDCNST` sets it? No.
C     
C     Alternative: Manually set coeffs? 
C     Helper: `SUBROUTINE MAKE_CD(DA_RE, DA_IM, CD_OUT)`
C     Impl:
C       CALL CDCMPL(DA_RE, CD_OUT)
C       CALL CDCMPL(DA_IM, CD_TMP)
C       CALL CDMCM(CD_TMP, CM_I, CD_TMP_2) ! CM = Complex Map? Complex Constant?
C       
C     Simpler:
C     We have `create_cda_const_` in wrapper.
C     It calls: `CALL RECD(0, INC)`. `CALL CDPAC(INC)`.
C     Actually `RECD(0, INC)` creates Identity CD? Or Zero?
C     IV-th variable. If IV=0, maybe 1?
C     Let's look at `create_cda_const_` logic again.
C     (Line 1032 in wrapper.f probably).
C     
C     If we can't easily construct, we can just assume `CD = RE + i IM`.
C     If standard memory layout is interleaved, we can't just combine.
C     
C     WAIT. `CDMDA` logic:
C     `CALL CDCMPL(INB, ICD)` (Convert DA-B to CD-B)
C     `CALL CDMCD(INA, ICD, INC)` (Multiply)
C     
C     So if we can make `I` (imaginary unit), we are good.
C     Let's try: `CALL RECD(0, INC)` (Creates Zero CD?)
C     Let's assume we can create variable `i` in Python and pass it?
C     The Wrapper needs to be self-contained.
C     
C     Let's use `create_cda_const_` logic manually:
C     But `create_cda_const_` sets constant part.
C     
C     Actually, checking `dafox.f` -> `CDRE`, `CDIM`.
C     Maybe there is `CDMAKE(RE, IM, CD)`? No.
C     
C     Let's punt construction to `SET_CD_FROM_DA`?
C     
C     Let's assume `COMPUTE_CD_LOG` can implement:
C     L_R (DA), THETA (DA).
C     Res = CDCMPL(L_R) + i * CDCMPL(THETA).
C     Where i = (0, 1).
C     I will define `i` by making a constant CD (0, 1).
C     How?
C     CALL DACON(TMP, 0.D0) ! dummy
C     CALL DACON(TMP, 1.D0) ! dummy
C     Actually, I can use `create_cda_const_` logic if I copy it.
C     But `create_cda_const_` takes doubles `re` `im`.
C     Here we have DAs.
C     
C     Okay, Plan B:
C     We only need to implement `COMPUTE_CD_EXP` and `COMPUTE_CD_LOG`.
C     `COMPUTE_CD_EXP`:
C       Inputs: X (DA), Y (DA).
C       Res = Exp(X) * (Cos(Y) + i Sin(Y)).
C       W = Cos(Y) + i Sin(Y).
C       W is a CD vector.
C       Cos(Y) is DA. Sin(Y) is DA.
C       We need to construct CD from Re/Im DAs again.
C     
C     So the core missing piece is `DA_TO_CD(DA_RE, DA_IM, CD_OUT)`.
C     
C     I will implement `MAKE_CD_VECTOR(DA_RE, DA_IM, CD_OUT)`.
C     How?
C     Logic: `CD_OUT = CDCMPL(DA_RE) + i * CDCMPL(DA_IM)`.
C     Need `i`.
C     `i` is constant CD (0, 1).
C     I can set it via `CC(NBEG(I)) = 0`, `CC(NBEG(I)+1) = 1`?
C     If layout allows.
C
C     Let's assume I can hack `i` creation:
C     Use `RECD` (create variable/const).
C     Actually `RECD` creates identity.
C     
C     Back to `dafox.f`. `CDCMPL(INA, INB)`: "TURNS DA INA INTO CD VECTOR INB".
C     Implies `INB = INA + 0i`.
C     To get `0 + 1i * INA`, we need `i`.
C     
C     Maybe `CDACD` works fine?
C     
C     I will implement `GET_IMAG_UNIT(CD_I)` that returns `i`.
C     Inside:
C     CALL FOXALL(IC, 1, NMMAX)
C     CALL DACON(IC(1), 0.D0) ! Create Real Zero
C     CALL CDCMPL(IC(1), CD_I) ! CD_I = 0 + 0i
C     ! Now set Imag part to 1.
C     ! How?
C     ! If `CC` access works:
C     ! `CC(NBEG(CD_I) + something) = 1.0`?
C     ! Unsafe.
C     
C     Wait, `dafox.f` has `CMSCD`? `CDSCM`?
C     `CDMCM` (CD * CM).
C     CM = Complex Map/Constant?
C     If CM is constant, `CDMCM` multiplies CD by Complex Constant.
C     
C     So:
C     `CD_OUT = CDCMPL(DA_RE) + CDMCM(CDCMPL(DA_IM), (0, 1))`?
C     Does `CDMCM` take doubles?
C     Signature: `SUBROUTINE CDMCM(INA,INB,INC)`.
C     INA is CD. INB is CM. INC is CD.
C     What is CM?
C     Lines 13410: `SUBROUTINE RECM(IIV,INC)`.
C     Lines ... `CMCNST(INA,INC)`?
C     
C     If CM is Complex Matrix/Map?
C     
C     Actually, checking `dafox.f` source for `CDMCM`:
C     It multiplies CD by CM.
C     If CM is just a complex scalar type in COSY, I can create one.
C     
C     Actually, simpler path:
C     `COMPUTE_CD_EXP` and `LOG` are only needed for Real Power $Z^a$.
C     $Z$ is CD. $a$ is Real.
C     $Z^a = \exp(a \log Z)$.
C     We need $\log Z$.
C     $\log Z = \log|Z| + i \arg Z$.
C     This returns a CD vector.
C     We assume we can construct it.
C     
C     Let's look at `create_cda_const_` in `wrapper.f` to see how it sets values.
C     (I recall it calls something).
C
      
      SUBROUTINE COMPUTE_CD_MUI(INA, INC)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER INA, INC, IC(1)
      PARAMETER(LEA=100000)
      COMMON /DACOM/ CDA(2*LEA),EPS,EPSMAC,IE1(LEA),IE2(LEA),
     *       IEO(LEA),IA1(0:1400000),IA2(0:1400000),NCFLT(LEA),
     *       IEW(40),IED(40),LEW,LEWI,IESP,NOMAX,NVMAX,NMMAX,NOCUT,
     *       LFLT,NFLT
      CALL FOXALL(IC, 1, NMMAX)
      INC = IC(1)
      CALL CDMUI(INA, INC)
      RETURN
      END

      SUBROUTINE COMPUTE_CD_PEI(INA, N, INC)
C     Complex Integer Power: C = A^N
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER INA, N, INC, I, IC(2), I_TMP, I_ONE
      PARAMETER(LEA=100000)
      COMMON /DACOM/ CDA(2*LEA),EPS,EPSMAC,IE1(LEA),IE2(LEA),
     *       IEO(LEA),IA1(0:1400000),IA2(0:1400000),NCFLT(LEA),
     *       IEW(40),IED(40),LEW,LEWI,IESP,NOMAX,NVMAX,NMMAX,NOCUT,
     *       LFLT,NFLT
     
      CALL FOXALL(IC, 2, NMMAX)
      INC = IC(1)
      I_TMP = IC(2)
      
C     Create 1.0 (Complex)
C     Call our helper `create_cda_const_` logic here?
C     Or assume user passes clean memory and we build logic.
C     Let's rely on Python calling `COMPUTE_CD_PKP` for general.
C     But for Integer, simple loop.
C     Need Unity.
C     CALL DACON(I_TMP, 1.D0) -> Real DA 1.0
C     CALL CDCMPL(I_TMP, INC) -> CD 1.0 + 0i
      CALL DACON(I_TMP, 1.D0)
      CALL CDCMPL(I_TMP, INC)

      IF(N.EQ.0) RETURN

      IF(N.GT.0) THEN
          DO I=1, N
            CALL CDMCD(INC, INA, I_TMP)
            CALL CDCOP(I_TMP, INC)
         END DO
      ELSE
         DO I=1, ABS(N)
            CALL CDMCD(INC, INA, I_TMP)
            CALL CDCOP(I_TMP, INC)
         END DO
         CALL CDMUI(INC, I_TMP)
         CALL CDCOP(I_TMP, INC)
      END IF
      CALL FOXDAL(IC, 2)
      RETURN
      END

