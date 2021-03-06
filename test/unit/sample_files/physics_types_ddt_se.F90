module physics_types_ddt

  use ccpp_kinds, only: kind_phys

  implicit none
  private

!> \section arg_table_physics_state  Argument Table
!! \htmlinclude physics_state.html
  type, public :: physics_state
    ! ncol: Number of horizontal columns
    integer                          :: ncol = 0
    ! latitude: Latitude
    real(kind_phys),         pointer :: latitude(:) => NULL()
  end type physics_state

!> \section arg_table_physics_types_ddt  Argument Table
!! \htmlinclude physics_types_ddt.html
  ! longitude: Longitude
  real(kind_phys),     public, pointer, protected :: longitude(:) => NULL()
  ! phys_state: Physics state variables updated by dynamical core
  type(physics_state), public                   :: phys_state

!! public interfaces
  public :: allocate_physics_types_ddt_fields

CONTAINS

  subroutine allocate_physics_types_ddt_fields(horizontal_dimension, set_init_val_in,             &
       reallocate_in)
    use shr_infnan_mod,   only: nan => shr_infnan_nan, assignment(=)
    use cam_abortutils,   only: endrun
    !! Dummy arguments
    integer,           intent(in) :: horizontal_dimension
    logical, optional, intent(in) :: set_init_val_in
    logical, optional, intent(in) :: reallocate_in

    !! Local variables
    logical                     :: set_init_val
    logical                     :: reallocate
    character(len=*), parameter :: subname = "allocate_physics_types_ddt_fields"

    ! Set optional argument values
    if (present(set_init_val_in)) then
      set_init_val = set_init_val_in
    else
      set_init_val = .true.
    end if
    if (present(reallocate_in)) then
      reallocate = reallocate_in
    else
      reallocate = .false.
    end if

    if (associated(longitude)) then
      if (reallocate) then
        deallocate(longitude)
        nullify(longitude)
      else
        call endrun(subname//": longitude is already associated, cannot allocate")
      end if
    end if
    allocate(longitude(horizontal_dimension))
    if (set_init_val) then
      longitude = nan
    end if
    if (set_init_val) then
      phys_state%ncol = 0
    end if
    if (associated(phys_state%latitude)) then
      if (reallocate) then
        deallocate(phys_state%latitude)
        nullify(phys_state%latitude)
      else
        call endrun(subname//": phys_state%latitude is already associated, cannot allocate")
      end if
    end if
    allocate(phys_state%latitude(horizontal_dimension))
    if (set_init_val) then
      phys_state%latitude = nan
    end if
  end subroutine allocate_physics_types_ddt_fields

end module physics_types_ddt
