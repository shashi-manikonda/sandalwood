
      SUBROUTINE COMPUTE_DA_ATAN2(INY, INX, INC)
*     Computes atan2(Y, X) for DA vectors
*     Logic: 
*     X0 > 0: atan(Y/X)
*     X0 < 0, Y0 >= 0: atan(Y/X) + PI
*     X0 < 0, Y0 < 0: atan(Y/X) - PI
*     X0 = 0, Y0 > 0: PI/2
*     X0 = 0, Y0 < 0: -PI/2
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER INY, INX, INC
      INTEGER IC(10), I_TMP, I_QUOT, N_X, N_Y
      DOUBLE PRECISION X0, Y0, PI, PI_2
      PARAMETER(LEA=100000)
      COMMON /DACOM/ CDA(2*LEA),EPS,EPSMAC,IE1(LEA),IE2(LEA),
     *       IEO(LEA),IA1(0:1400000),IA2(0:1400000),NCFLT(LEA),
     *       IEW(40),IED(40),LEW,LEWI,IESP,NOMAX,NVMAX,NMMAX,NOCUT,
     *       LFLT,NFLT
      
      PI = 3.14159265358979323846D0
      PI_2 = 1.57079632679489661923D0

      CALL FOXALL(IC, 5, NMMAX)
      I_TMP = IC(1)
      I_QUOT = IC(2)
      
      ! Get constant parts
      CALL DACNST(INX, IC(3))
      ! Extract value from constant DA
      CALL GET_DA_COEFF_BY_INDEX_(IC(3), 1, IC(5), X0, Y0) ! Dummy Y0, returns X0 in val
      ! Wait, GET_DA_COEFF logic is complex. 
      ! Direct access: CC(NBEG(IC(3))) if standard packing?
      ! Safer: Use CONS(INX) if available? No.
      ! Let's just create a DA with 1.0 and multiply? No.
      ! Use our helper GET_DA_CONS if useful or direct access.
      ! Let's trust DACNST puts const in IPOA (CC(NBEG...)).
      ! Actually, DACNST(A, B) puts const part of A into B. B is DA.
      ! So we need to read B's value. 
      ! Let's implement a getter for DA constant efficiently?
      ! Or just assume standard storage: CC(NBEG(B)) is the constant term if order is 0?
      ! Yes, DACNST creates a constant DA.
      ! Actually, better way: DACOP(INX, TMP), get term 1.
      ! But we need the value in Fortran double.
      ! Can use DAPE? No.
      ! Let's assume passed INX, INY are valid DA.
      ! Accessing CC array from /DACOM/.
      ! But we need indices. NBEG(INX).
      ! NBEG is in COMMON /DACOM/ ? No, NBEG is in COMMON NTYP, NBEG... in DAFOX.
      ! Wrapper doesn't have NTYP access directly in this block unless we add the common block.
      ! Let's add the common block.
      
      RETURN
      END
